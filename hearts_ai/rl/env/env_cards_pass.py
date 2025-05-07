import warnings
from typing import TypeAlias, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.engine import Card, HeartsCore, HeartsRules
from .obs import (
    create_cards_pass_env_obs,
    create_cards_pass_env_action_masks,
    create_play_env_obs_from_hearts_core,
    create_play_env_action_masks_from_hearts_core,
)
from .utils import (
    action_to_card,
    ActionTakingCallbackParam,
    handle_action_taking_callback_param,
)

PlayEnvObsType: TypeAlias = ObsType
PlayEnvActType: TypeAlias = ActType


class HeartsCardsPassEnvironment(gym.Env):
    """
    Environment for agents learning the card passing in Hearts.

    The action space is a discrete number from 0 to 51 inclusive, representing
    the index of a card to play. The cards are ordered in the following way:
    clubs, diamonds, spades, hearts, with ranks ordered from 2 to Ace.

    Each observation is a vector of length 52, encoding agent's relation to
    each card (-1: picked to pass / 0: not in hand / 1: in hand)

    Note that for some observation space types the environment provides
    information on cards that each player has. The algorithms need to take
    this into account and when evaluating, determinisation techniques need
    to be used.

    Agent picks cards one by one, and once the third card is picked, the
    agent receives a reward and the environment terminates. The reward is
    calculated as the average gain in points collected across simulated rounds.
    The number of simulations can be modified through ``eval_count``
    parameter. The round will be played out with the selected cards passed
    ``eval_count`` times, and the same number of times without card passing.
    Note that when calculating average gain, rounds where the player has
    shot the moon are values as -26.

    Args:
        opponents_callbacks: Callbacks responsible for decision making of other
            players. In a self-play environment, these should be snapshots of
            the learning agent. It should be either a list of 3 elements, each
            corresponding to each opponent clockwise, or a single object, in
            which case it is shared for all opponents. A callback should
            be a function accepting a state and an action mask, and returning
            an action.
        playing_callbacks: Callbacks used when performing rounds simulations.
            If it is a list, the first callback is responsible for the learning
            agent gameplay, and the rest for the remaining players clockwise.
            If it is a single callback, it is shared for every player.
            A callback should be a function accepting a play environment
            state and an action mask, and returning a play environment action.
        eval_count: How many times to simulate the round for each variant
            (with and without card passing). It can be a two-element list,
            specifying evaluation count separately for each variant, or a
            single integer, in which case it is identical for both.
        supress_deterministic_eval_warn: If set to False, the environment will
            inform the user when evaluations for either variant are identical
            10 times in a row.
    """

    def __init__(
            self,
            opponents_callbacks: ActionTakingCallbackParam[ObsType, ActType],
            playing_callbacks: ActionTakingCallbackParam[PlayEnvObsType, PlayEnvActType],
            eval_count: int | list[int] = 10,
            supress_deterministic_eval_warn: bool = False,
    ):
        super().__init__()

        self.opponents_callbacks = handle_action_taking_callback_param(opponents_callbacks, 3)
        self.playing_callbacks = handle_action_taking_callback_param(playing_callbacks, 4)

        if isinstance(eval_count, int):
            self.eval_count = [eval_count] * 2
        elif isinstance(eval_count, list) and len(eval_count) == 2:
            self.eval_count = eval_count
        else:
            raise ValueError('Unsupported eval_count value. It needs to be either an integer,'
                             'or a two-element list.')

        self.supress_deterministic_eval_warn = supress_deterministic_eval_warn
        self.__times_consecutive_eval_identical = [0, 0]

        self.action_space = gym.spaces.Discrete(52)
        self.observation_space = gym.spaces.Box(low=-1, high=26, shape=(52,), dtype=np.int8)

        # properly set in reset()
        self._hands: list[list[Card]] | None = None
        self.picked_cards: list[Card] = []

    @property
    def hands(self) -> list[list[Card]]:
        return [hand.copy() for hand in self._hands]

    def _get_obs(self) -> ObsType:
        return create_cards_pass_env_obs(
            player_hand=self.hands[0],
            picked_cards=self.picked_cards,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        core = HeartsCore(
            random_state=self.np_random.integers(999999),
        )
        core.next_round()
        self._hands = core.hands
        self.picked_cards = []

        return self._get_obs(), {}

    def __simulate_round(self, include_card_passing: bool) -> SupportsFloat:
        """
        Returns:
            Points for the round for the player at index 0, or -26 if the
            player has shot the moon
        """
        hearts_core = HeartsCore(
            rules=HeartsRules(passing_cards=include_card_passing),
            random_state=self.np_random.integers(999999),
        )
        hearts_core.next_round()
        hearts_core.hands = self.hands

        if include_card_passing:
            hearts_core.pick_cards_to_pass(0, self.picked_cards)

            for opponent_player_idx, opponent_callback in enumerate(self.opponents_callbacks, 1):
                opponent_hand = self.hands[opponent_player_idx]
                opponent_picked_cards = []
                for _ in range(3):
                    # opponents card pass
                    obs = create_cards_pass_env_obs(opponent_hand, opponent_picked_cards)
                    action_masks = create_cards_pass_env_action_masks(
                        opponent_hand, opponent_picked_cards)
                    action = opponent_callback(obs, action_masks)
                    card_to_pass = action_to_card(action)
                    opponent_picked_cards.append(card_to_pass)

                hearts_core.pick_cards_to_pass(opponent_player_idx, opponent_picked_cards)
            hearts_core.complete_pass_cards()

        while not hearts_core.is_round_finished:
            # simulate gameplay
            obs = create_play_env_obs_from_hearts_core(hearts_core)
            action_masks = create_play_env_action_masks_from_hearts_core(hearts_core)
            playing_callback = self.playing_callbacks[hearts_core.current_player_idx]
            action = playing_callback(obs, action_masks)
            card_to_play = action_to_card(action)
            hearts_core.play_card(card_to_play)

            if hearts_core.is_current_trick_full:
                hearts_core.complete_trick()

        end_of_round_score = hearts_core.current_round_scores[0]
        if hearts_core.is_moon_shot_triggered and end_of_round_score == 0:
            return -26
        return end_of_round_score

    def __handle_deterministic_eval_check(
            self,
            pts_with_passing: np.ndarray,
            pts_no_passing: np.ndarray,
    ):
        for i, pts in enumerate([pts_with_passing, pts_no_passing]):
            if np.all(pts == pts[0]):
                self.__times_consecutive_eval_identical[i] += 1
            else:
                self.__times_consecutive_eval_identical[i] = 0

        if self.__times_consecutive_eval_identical[0] == 10:
            warnings.warn(
                'During reward calculation, all evaluations for a variant with '
                'card passing were identical 10 times in a row. This might mean '
                'that card passing and playing decisions are deterministic. '
                'Consider setting the first element for ``eval_count`` parameter'
                'to 1 for faster performance.')
        if self.__times_consecutive_eval_identical[1] == 10:
            warnings.warn(
                'During reward calculation, all evaluations for a variant without '
                'card passing were identical 10 times in a row. This might mean '
                'that playing decisions are deterministic. '
                'Consider setting the second element for ``eval_count`` parameter'
                'to 1 for faster performance.')

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        card_to_pass = action_to_card(action)
        if card_to_pass not in self.hands[0]:
            warnings.warn('Illegal action - the selected card is not on hand.'
                          'Environment state will not change')
            return self._get_obs(), 0, False, False, {}

        self.picked_cards.append(card_to_pass)
        if len(self.picked_cards) < 3:
            return self._get_obs(), 0, False, False, {}

        pts_with_passing = np.array([
            self.__simulate_round(include_card_passing=True)
            for _ in range(self.eval_count[0])
        ])
        pts_no_passing = np.array([
            self.__simulate_round(include_card_passing=False)
            for _ in range(self.eval_count[1])
        ])

        if not self.supress_deterministic_eval_warn:
            self.__handle_deterministic_eval_check(pts_with_passing, pts_no_passing)

        reward = pts_with_passing.mean() - pts_no_passing.mean()
        return self._get_obs(), reward, True, False, {}

    def action_masks(self) -> list[bool]:
        return create_cards_pass_env_action_masks(self.hands[0], self.picked_cards)
