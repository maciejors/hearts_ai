from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from gymnasium.core import ObsType, ActType
from sb3_contrib import MaskablePPO

from hearts_ai.engine import Card, PassDirection, Suit, HeartsRound
from hearts_ai.engine.constants import PLAYER_COUNT, CARDS_TO_PASS_COUNT
from hearts_ai.engine.round import (
    STATE_IDX_TRICK_STARTING_PLAYER,
    STATE_IDX_ARE_HEARTS_BROKEN,
)
from hearts_ai.engine.utils import get_winning_card_argmax, points_for_card
from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.env import HeartsPlayEnvironment
from hearts_ai.rl.env.obs import (
    create_play_env_obs_full,
    create_cards_pass_env_obs,
    create_play_env_action_masks,
    create_cards_pass_env_action_masks,
    play_env_observation_settings,
)
from hearts_ai.rl.training.common import update_self_play_clones
from .base import BasePlayer
from ..deck import Deck
from ..utils import card_list_to_array, array_to_card_list, get_valid_plays_objs


class AgentWrapper(ABC):
    @abstractmethod
    def predict(
            self,
            obs: ObsType,
            action_masks: np.ndarray,
            round_state: HeartsRound | None = None,
    ) -> ActType:
        raise NotImplementedError()


class PPOWrapper(AgentWrapper):
    def __init__(self, agent: MaskablePPO):
        self.agent = agent

    def predict(
            self,
            obs: ObsType,
            action_masks: np.ndarray,
            round_state: HeartsRound | None = None,
    ) -> ActType:
        pred, _ = self.agent.predict(obs, action_masks=action_masks)
        return pred.item()


class MCTSRLWrapper(AgentWrapper):
    def __init__(self, agent: MaskableMCTSRL):
        self.agent = agent
        self.env = HeartsPlayEnvironment(
            opponents_callbacks=[],
            reward_setting='sparse'  # does not matter here
        )
        self.agent.set_env(self.env)
        update_self_play_clones(self.env, self.agent)

    def predict(
            self,
            obs: ObsType,
            action_masks: np.ndarray,
            round_state: HeartsRound | None = None,
    ) -> ActType:
        assert round_state is not None
        self.env.round = round_state
        pred, _ = self.agent.predict(np.array([obs]), action_masks=None, sb3_eval_mode=False)
        return pred.item()


def _opponents_voids_default_factory() -> dict[Suit, list[bool]]:
    result = {}
    for suit in Suit:
        result[suit] = [False] * (PLAYER_COUNT - 1)
    return result


class RLPlayer(BasePlayer):
    @dataclass
    class Memory:
        curr_trick_no: int = 0
        # the suits the opponents have voided (True - voided, False - might have this suit)
        opponents_voids: dict[Suit, list[bool]] = field(default_factory=_opponents_voids_default_factory)
        cards_played_by_each: list[list[Card]] = field(default_factory=lambda: [[] for _ in range(PLAYER_COUNT)])
        points: list[int] = field(default_factory=lambda: [0] * PLAYER_COUNT)
        pass_direction: PassDirection = PassDirection.NO_PASSING
        passed_cards_in_play: list[Card] = field(default_factory=list)
        my_idx_in_curr_trick: int | None = None

    def __init__(
            self,
            playing_agent: AgentWrapper,
            card_passing_agent: AgentWrapper,
            play_env_obs_setting: Literal['full', 'compact'],
            simulation_count: int = 20,
            random_state: int | None = None,
    ):
        self.playing_agent = playing_agent
        self.card_passing_agent = card_passing_agent
        self.create_obs = play_env_observation_settings[play_env_obs_setting]
        self.simulation_count = simulation_count
        self.memory = RLPlayer.Memory()
        self._np_random = np.random.default_rng(random_state)

    def _get_indexes_of_players_in_trick(self) -> list[int]:
        """
        Returns:
            indexes of players in the list trick, relative to us
            (0 - us, 1 - left, 2 - across, 3 - right)
        """
        return list(np.array(range(PLAYER_COUNT)) - self.memory.my_idx_in_curr_trick)

    def play_card(
            self,
            hand: list[Card],
            trick: list[Card],
            are_hearts_broken: bool,
            is_first_trick: bool
    ) -> Card:
        self.memory.my_idx_in_curr_trick = len(trick)
        self.memory.curr_trick_no += 1
        leading_suit = None

        if len(trick) > 0:
            leading_suit = trick[0].suit
            for player_idx, card_played in zip(self._get_indexes_of_players_in_trick()[:len(trick)], trick):
                if player_idx != 0 and card_played.suit != leading_suit:
                    self.memory.opponents_voids[leading_suit][player_idx - 1] = True

        valid_plays = get_valid_plays_objs(hand, leading_suit, are_hearts_broken, is_first_trick)
        if len(valid_plays) == 1:
            return valid_plays[0]

        votes = []
        success_count = 0
        failures_count = 0
        for _ in range(self.simulation_count):
            act_masks = create_play_env_action_masks(
                card_list_to_array(hand),
                Suit.order(leading_suit),
                are_hearts_broken,
                is_first_trick,
            )
            # skip malformed determinisations which cause problems
            max_attempts = 20
            for _ in range(max_attempts):
                try:
                    determinisation_round = self._determinisation_round(hand, trick, are_hearts_broken)
                    act = self.playing_agent.predict(
                        obs=self.create_obs(determinisation_round),
                        action_masks=act_masks,
                        round_state=determinisation_round,
                    )
                    votes.append(act)
                    success_count += 1
                    break
                except RuntimeError:
                    failures_count += 1

        determinisation_success_rate = success_count / (success_count + failures_count) * 100
        if determinisation_success_rate < 1e-10 or len(votes) == 0:
            print(f'All determinisations were unsuccessful. Playing the first available card')
            return valid_plays[0]
        if determinisation_success_rate < 70:
            print(f'Low determinisation success rate occurred ({determinisation_success_rate:.2f}%)')

        return Card(Counter(votes).most_common(1)[0][0])

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        self.memory.pass_direction = direction
        cards_to_pass = np.array([], dtype=np.int16)
        for _ in range(CARDS_TO_PASS_COUNT):
            obs = create_cards_pass_env_obs(
                card_list_to_array(hand),
                cards_to_pass,
                direction,
            )
            act_masks = create_cards_pass_env_action_masks(
                card_list_to_array(hand),
                cards_to_pass,
            )
            act = self.card_passing_agent.predict(obs, act_masks)
            cards_to_pass = np.append(cards_to_pass, act)

        cards_to_pass_objs = array_to_card_list(cards_to_pass)
        self.memory.passed_cards_in_play = cards_to_pass_objs.copy()
        return cards_to_pass_objs

    def post_trick_callback(self, trick: list[Card], is_trick_taken: bool) -> None:
        indexes_of_players_in_trick = self._get_indexes_of_players_in_trick()
        winner_idx = get_winning_card_argmax(
            card_list_to_array(trick),
            Suit.order(trick[0].suit),
        )
        my_idx = indexes_of_players_in_trick[0]
        winner_idx_relative_to_me = (winner_idx - my_idx) % 4
        pts_in_trick = sum(points_for_card(c.idx) for c in trick)
        self.memory.points[winner_idx_relative_to_me] += pts_in_trick

        leading_suit = trick[0].suit
        for player_idx, card_played in zip(indexes_of_players_in_trick, trick):
            self.memory.cards_played_by_each[player_idx].append(card_played)
            if player_idx != 0 and card_played.suit != leading_suit:
                self.memory.opponents_voids[leading_suit][player_idx - 1] = True
            if card_played in self.memory.passed_cards_in_play:
                self.memory.passed_cards_in_play.remove(card_played)

    def post_round_callback(self, score: int) -> None:
        self.memory = RLPlayer.Memory()

    def _determinisation_round(self, my_hand: list[Card], trick: list[Card], are_hearts_broken: bool) -> HeartsRound:
        """Create a HeartsRound object based on the known information"""
        all_cards = set(Deck(random_state=int(self._np_random.integers(999999))).deal(52))
        seen_cards = set(my_hand) | set(trick)
        for player_cards in self.memory.cards_played_by_each:
            seen_cards.update(player_cards)
        seen_cards.update(self.memory.passed_cards_in_play)

        cards_unknown_owner = list(all_cards - seen_cards)
        self._np_random.shuffle(cards_unknown_owner)

        # the player can just always assume he is at index 0
        hands = [my_hand.copy()] + [[] for _ in range(PLAYER_COUNT - 1)]
        opponent_indices = [1, 2, 3]

        # assign passed cards
        if self.memory.pass_direction != PassDirection.NO_PASSING:
            pass_to_idx_map = {
                PassDirection.LEFT: 1,
                PassDirection.RIGHT: 3,
                PassDirection.ACROSS: 2,
            }
            receiver_idx = pass_to_idx_map[self.memory.pass_direction]
            for card_in_trick in trick:
                if card_in_trick in self.memory.passed_cards_in_play:
                    self.memory.passed_cards_in_play.remove(card_in_trick)
            hands[receiver_idx].extend(self.memory.passed_cards_in_play)

        cards_needed_counts = [len(my_hand) - len(hand) for hand in hands]
        # players before us should have 1 less card
        for i in range(1, len(trick) + 1):
            cards_needed_counts[-i] -= 1

        # assign remaining cards
        for card in cards_unknown_owner:
            possible_receivers = []
            for i, opponent_idx in enumerate(opponent_indices):
                if cards_needed_counts[opponent_idx] == 0:
                    continue
                if not self.memory.opponents_voids[card.suit][i]:
                    possible_receivers.append(opponent_idx)

            if not possible_receivers:
                # fallback - give to anyone that still needs cards
                possible_receivers = [i for i in opponent_indices if cards_needed_counts[i] > 0]

            if not possible_receivers:
                raise RuntimeError('Determinisation failed')

            assigned = self._np_random.choice(possible_receivers)
            hands[assigned].append(card)
            cards_needed_counts[assigned] -= 1

        obs = create_play_env_obs_full(
            trick_no=self.memory.curr_trick_no,
            player_idx=0,
            trick_starting_player_idx=self._get_indexes_of_players_in_trick()[0],
            current_trick_ordered=card_list_to_array(trick),
            hands=[card_list_to_array(cl) for cl in hands],
            played_cards=[card_list_to_array(cl) for cl in self.memory.cards_played_by_each],
            current_round_points_collected=np.array(self.memory.points),
        )

        np_state = np.zeros(271, dtype=np.int16)
        np_state[:217] = np.copy(obs)

        if are_hearts_broken:
            np_state[STATE_IDX_ARE_HEARTS_BROKEN] = 1

        trick_starting_player_idx = -self.memory.my_idx_in_curr_trick % 4
        np_state[STATE_IDX_TRICK_STARTING_PLAYER] = trick_starting_player_idx

        hearts_round = HeartsRound(random_state=0)
        hearts_round.are_cards_passed = True
        # noinspection PyProtectedMember
        hearts_round._np_state = np_state
        return hearts_round
