import copy
import warnings
from dataclasses import dataclass, field
from typing import TypeAlias, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.engine import HeartsRound, HeartsRules, PassDirection
from hearts_ai.engine.constants import PLAYER_COUNT
from .obs import (
    create_cards_pass_env_obs,
    create_cards_pass_env_action_masks,
    create_play_env_obs_from_hearts_round,
    create_play_env_action_masks_from_hearts_round,
)
from .utils import (
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

    Each observation is a vector of length 55, the elements at indices 0-51
    encode agent's relation to each card (-1: picked to pass / 0: not in hand / 1: in hand),
    and at indices 52-54 one-hot encode the information on the pass direction
    (obs[52] == 1 -> left, 53 -> across, 54 -> right)

    Agent picks cards one by one, and once the third card is picked, the
    agent receives a reward and the environment terminates. The reward is
    calculated as the average gain in points collected across simulated rounds.
    The number of simulations can be modified through ``eval_count``
    parameter. The round will be played out with the selected cards passed
    ``eval_count`` times, and the same number of times without card passing.
    Note that when calculating average gain, rounds where the player has
    shot the moon are values as -26.

    Args:
        opponents_callbacks: Callbacks responsible for decision-making of other
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

    __slots__ = [
        'opponents_callbacks',
        'playing_callbacks',
        'action_space',
        'observation_space',
        'suppress_deterministic_eval_warn',
        'eval_count',
        'state',
        '__times_consecutive_eval_identical',
    ]

    @dataclass(slots=True)
    class State:
        _pass_direction: PassDirection
        _hands: list[np.ndarray]
        _picked_cards: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))

        @property
        def pass_direction(self) -> PassDirection:
            return self._pass_direction

        @property
        def hands(self) -> list[np.ndarray]:
            return self._hands

        @property
        def picked_cards(self) -> np.ndarray:
            return self._picked_cards

        def __deepcopy__(self, memo):
            state_copy = copy.copy(self)
            state_copy._hands = [card_arr.copy() for card_arr in self._hands]
            state_copy._picked_cards = self._picked_cards.copy()
            return state_copy

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

        self.suppress_deterministic_eval_warn = supress_deterministic_eval_warn
        self.__times_consecutive_eval_identical = [0, 0]

        self.action_space = gym.spaces.Discrete(52)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(55,), dtype=np.int8)

        # properly set in reset()
        self.state: HeartsCardsPassEnvironment.State | None = None

    def _get_obs(self) -> ObsType:
        return create_cards_pass_env_obs(
            player_hand=self.state.hands[0],
            picked_cards=self.state.picked_cards,
            pass_direction=self.state.pass_direction,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        hearts_round = HeartsRound(
            random_state=int(self.np_random.integers(999999)),
        )

        if self.state is None:
            next_pass_direction = PassDirection.LEFT
        else:
            next_pass_direction = list(PassDirection)[self.state.pass_direction.value + 1]
            if next_pass_direction == PassDirection.NO_PASSING:
                next_pass_direction = PassDirection.LEFT

        self.state = HeartsCardsPassEnvironment.State(
            _pass_direction=next_pass_direction,
            _hands=[hearts_round.get_hand(i) for i in range(PLAYER_COUNT)],
        )
        return self._get_obs(), {}

    def __simulate_round(self, include_card_passing: bool) -> SupportsFloat:
        """
        Returns:
            Points for the round for the player at index 0, or -26 if the
            player has shot the moon
        """
        hearts_round = HeartsRound(
            rules=HeartsRules(passing_cards=include_card_passing),
            random_state=int(self.np_random.integers(999999)),
            pass_direction=self.state.pass_direction
        )
        for player_idx in range(PLAYER_COUNT):
            hearts_round.override_hand(player_idx, self.state.hands[player_idx])

        if include_card_passing:
            hearts_round.pick_cards_to_pass(0, self.state.picked_cards)

            for opponent_player_idx, opponent_callback in enumerate(self.opponents_callbacks, 1):
                opponent_hand = self.state.hands[opponent_player_idx]
                opponent_picked_cards = np.array([], dtype=np.int16)
                for _ in range(3):
                    # opponents card pass
                    obs = create_cards_pass_env_obs(
                        opponent_hand, opponent_picked_cards, self.state.pass_direction
                    )
                    action_masks = create_cards_pass_env_action_masks(
                        opponent_hand, opponent_picked_cards
                    )
                    action = opponent_callback(obs, action_masks)
                    opponent_picked_cards = np.append(opponent_picked_cards, action)

                hearts_round.pick_cards_to_pass(opponent_player_idx, opponent_picked_cards)
            hearts_round.perform_cards_passing()

        while not hearts_round.is_finished:
            # simulate gameplay
            obs = create_play_env_obs_from_hearts_round(hearts_round)
            action_masks = create_play_env_action_masks_from_hearts_round(hearts_round)
            playing_callback = self.playing_callbacks[hearts_round.current_player_idx]
            action = playing_callback(obs, action_masks)
            if type(action) == np.ndarray:
                action = action.item()
            hearts_round.play_card(action)

            if hearts_round.is_current_trick_full:
                hearts_round.complete_trick()

        end_of_round_score = hearts_round.scores[0]
        if hearts_round.is_moon_shot_triggered and end_of_round_score == 0:
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
        if action not in self.state.hands[0]:
            warnings.warn('Illegal action - the selected card is not on hand.'
                          'Environment state will not change')
            return self._get_obs(), 0, False, False, {}

        if action in self.state.picked_cards:
            warnings.warn('Illegal action - the selected card is already picked.'
                          'Environment state will not change')
            return self._get_obs(), 0, False, False, {}

        self.state._picked_cards = np.append(self.state.picked_cards, action)
        if len(self.state.picked_cards) < 3:
            return self._get_obs(), 0, False, False, {}

        pts_with_passing = []
        pts_no_passing = []

        for _ in range(self.eval_count[0]):
            try:
                pts_with_passing.append(self.__simulate_round(include_card_passing=True))
            except Exception as err:
                warnings.warn(f'An error occurred while simulating a round with passing: {err}')

        for _ in range(self.eval_count[1]):
            try:
                pts_no_passing.append(self.__simulate_round(include_card_passing=False))
            except Exception as err:
                warnings.warn(f'An error occurred while simulating a round without passing: {err}')

        pts_with_passing = np.array(pts_with_passing)
        pts_no_passing = np.array(pts_no_passing)

        if not self.suppress_deterministic_eval_warn:
            self.__handle_deterministic_eval_check(pts_with_passing, pts_no_passing)

        reward = pts_no_passing.mean() - pts_with_passing.mean()
        return self._get_obs(), reward, True, False, {'is_success': reward > 0}

    def action_masks(self) -> np.ndarray:
        return create_cards_pass_env_action_masks(
            self.state.hands[0], self.state.picked_cards
        )

    def __deepcopy__(self, memo):
        env_copy = copy.copy(self)
        env_copy.__times_consecutive_eval_identical = self.__times_consecutive_eval_identical.copy()
        env_copy.state = copy.deepcopy(self.state)
        return env_copy
