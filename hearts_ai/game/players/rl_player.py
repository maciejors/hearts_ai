from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from hearts_ai.engine import Card, PassDirection, Suit
from hearts_ai.engine.constants import PLAYER_COUNT, CARDS_TO_PASS_COUNT
from hearts_ai.engine.utils import get_winning_card_argmax, points_for_card
from hearts_ai.rl.env.obs import (
    create_play_env_obs,
    create_cards_pass_env_obs,
    create_play_env_action_masks,
    create_cards_pass_env_action_masks,
)
from hearts_ai.rl.training.common import SupportedAlgorithm
from .base import BasePlayer
from ..deck import Deck
from ..utils import card_list_to_array, array_to_card_list


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
        passed_cards: list[Card] = field(default_factory=list)
        my_idx_in_curr_trick: int | None = None

    def __init__(
            self,
            playing_agent: SupportedAlgorithm,
            card_passing_agent: SupportedAlgorithm,
            simulation_count: int = 10,
            random_state: int | None = None,
    ):
        self.playing_agent = playing_agent
        self.card_passing_agent = card_passing_agent
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

        votes = []
        for _ in range(self.simulation_count):
            obs = self._determinisation_state(hand, trick)
            act_masks = create_play_env_action_masks(
                card_list_to_array(hand),
                Suit.order(leading_suit),
                are_hearts_broken,
                is_first_trick,
            )
            act = self.playing_agent.predict(obs, action_masks=act_masks)[0]
            votes.append(act)

        return Card(Counter(votes).most_common(1)[0][0])

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        self.memory.pass_direction = direction
        cards_to_pass = np.array([])
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
            act = self.card_passing_agent.predict(obs, action_masks=act_masks)[0]
            cards_to_pass = np.append(cards_to_pass, act)

        cards_to_pass_objs = array_to_card_list(cards_to_pass)
        self.memory.passed_cards = cards_to_pass_objs.copy()
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
        self.memory.points[winner_idx_relative_to_me] = pts_in_trick

        leading_suit = trick[0].suit
        for player_idx, card_played in zip(indexes_of_players_in_trick, trick):
            self.memory.cards_played_by_each[player_idx].append(card_played)
            if player_idx != 0 and card_played.suit != leading_suit:
                self.memory.opponents_voids[leading_suit][player_idx - 1] = True

    def post_round_callback(self, score: int) -> None:
        self.memory = RLPlayer.Memory()

    def _determinisation_state(self, hand: list[Card], trick: list[Card]) -> np.ndarray:
        """Form the current state using the known information as well as determinisation"""
        all_cards = set(Deck(random_state=int(self._np_random.integers(999999))).deal(52))
        seen_cards = set(hand) | set(trick)
        for player_cards in self.memory.cards_played_by_each:
            seen_cards.update(player_cards)
        seen_cards.update(self.memory.passed_cards)

        cards_unknown_owner = list(all_cards - seen_cards)
        self._np_random.shuffle(cards_unknown_owner)

        hands = [hand] + [[] for _ in range(PLAYER_COUNT - 1)]
        opponent_indices = [1, 2, 3]

        # assign passed cards
        if self.memory.pass_direction != PassDirection.NO_PASSING:
            pass_to_idx_map = {
                PassDirection.LEFT: 1,
                PassDirection.RIGHT: 3,
                PassDirection.ACROSS: 2,
            }
            receiver_idx = pass_to_idx_map[self.memory.pass_direction]
            hands[receiver_idx].extend(self.memory.passed_cards)

        # assign remaining cards
        for card in cards_unknown_owner:
            possible_receivers = []
            for i, opponent_idx in enumerate(opponent_indices):
                if not self.memory.opponents_voids[card.suit][i]:
                    possible_receivers.append(opponent_idx)
            if not possible_receivers:
                # in case something goes wrong just give the card to anyone
                # we do not want this to be too complex computationally
                possible_receivers = opponent_indices
            assigned = self._np_random.choice(possible_receivers)
            hands[assigned].append(card)

        obs = create_play_env_obs(
            trick_no=self.memory.curr_trick_no,
            player_idx=0,
            trick_starting_player_idx=self._get_indexes_of_players_in_trick()[0],
            current_trick_ordered=card_list_to_array(trick),
            hands=[card_list_to_array(cl) for cl in hands],
            played_cards=[card_list_to_array(cl) for cl in self.memory.cards_played_by_each],
            current_round_points_collected=np.array(self.memory.points),
        )
        return obs
