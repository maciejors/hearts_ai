import warnings
from typing import Literal, Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.engine import HeartsCore, HeartsRules
from hearts_ai.engine.utils import points_for_card
from .obs import (
    create_play_env_obs_from_hearts_core,
    create_play_env_action_masks_from_hearts_core,
)
from .utils import (
    action_to_card,
    ActionTakingCallbackParam,
    handle_action_taking_callback_param,
)


class HeartsPlayEnvironment(gym.Env):
    """
    Environment for agents learning to play the engine of hearts
    (without passing cards).

    The action space is a discrete number from 0 to 51 inclusive, representing
    the index of a card to play. The cards are ordered in the following way:
    clubs, diamonds, spades, hearts, with ranks ordered from 2 to Ace.

    Each observation is a vector of length 217, containing the
    following information (by indices):

        0: Trick number
        1-52: Agent's relation to each card
            (-1: played before, 0: not in hand, 1: in hand)
        53-104: Left opponent's relation to each card
            (-1: played before, 0: not in hand, 1: in hand, 2: played in this trick)
        105-156: Across opponent's relation to each card
        157-208: Right opponent's relation to each card
        209-212: One-hot vector specifying the leading suit in this trick
        213-216: Current round score of each player (agent, left, across, right)

    Note that for some observation space types the environment provides
    information on cards that each player has. The algorithms need to take
    this into account and when evaluating, determinisation techniques need
    to be used.

    There are two reward settings: sparse and dense. In the sparse reward
    setting, a reward is only given at the end of a round. This reward is
    calculated as the negative of the number of points received after
    the round, or, if the agent has shot the moon, the reward is set to 26.

    In the dense reward setting, a reward is given after each trick. The value
    of this reward is equal to the number of points within the trick if the
    agent avoided taking this trick, or the negative of it if the agent has
    taken the trick. However, if the agent has managed to shoot the moon, the
    reward for the last trick in the round is always 26 regardless of the number
    of points collected in that trick, and if another player has shot the moon,
    the reward is -26.

    Args:
        opponents_callbacks: Callbacks responsible for decision making of other
            players. In a self-play environment, these should represent
            snapshots of the learning agent. This parameter should be a list of
            3 functions, each one accepting a state and an action mask, and
            returning an action. Alternatively it could also be a single object,
            in which case it will be shared for all opponents.
        reward_setting: Whether to use the sparse or dense reward setting.
            Default: dense
    """

    def __init__(self,
                 opponents_callbacks: ActionTakingCallbackParam[ObsType, ActType],
                 reward_setting: Literal['dense', 'sparse'] = 'dense'):
        super().__init__()

        self.opponents_callbacks = handle_action_taking_callback_param(opponents_callbacks, 3)

        allowed_reward_settings = ['dense', 'sparse']
        if reward_setting not in allowed_reward_settings:
            raise ValueError(f'Invalid `reward_setting`: "{reward_setting}". '
                             f'Accepted values: {allowed_reward_settings}.')
        self.reward_setting = reward_setting

        self.action_space = gym.spaces.Discrete(52)
        self.observation_space = gym.spaces.Box(low=-1, high=26, shape=(217,), dtype=np.int8)

        # properly set in reset()
        self.core: HeartsCore | None = None

    def _get_obs(self) -> ObsType:
        return create_play_env_obs_from_hearts_core(self.core)

    def __simulate_next_opponent(self):
        opponent_callback = self.opponents_callbacks[self.core.current_player_idx - 1]
        opponent_action = opponent_callback(self._get_obs(), self.action_masks())
        opponent_card = action_to_card(opponent_action)
        self.core.play_card(opponent_card)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.core = HeartsCore(
            rules=HeartsRules(passing_cards=False),
            random_state=self.np_random.integers(999999),
        )

        self.core.next_round()
        while self.core.current_player_idx != 0:
            self.__simulate_next_opponent()

        return self._get_obs(), {}

    @staticmethod
    def _calculate_dense_reward(
            is_round_finished: bool,
            current_round_points_collected: list[int],
            trick_points: int,
            is_trick_taken: bool,
    ) -> int:
        # the agent has shot the moon
        if is_round_finished and current_round_points_collected[0] == 26:
            return 26
        # someone else has shot the moon
        if is_round_finished and any([p == 26 for p in current_round_points_collected]):
            return -26

        sign = -2 * is_trick_taken + 1
        reward = sign * trick_points
        return reward

    @staticmethod
    def _calculate_sparse_reward(
            is_round_finished: bool,
            current_round_points_collected: list[int]
    ) -> int:
        if not is_round_finished:
            return 0
        if current_round_points_collected[0] == 26:
            return 26
        if any([p == 26 for p in current_round_points_collected]):
            return -26
        return -current_round_points_collected[0]

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # take an action
        card_to_play = action_to_card(action)
        if card_to_play not in self.core.hands[0]:
            warnings.warn('Illegal action - the selected card is not on hand.'
                          'Environment state will not change')
            return self._get_obs(), 0, False, False, {}

        self.core.play_card(card_to_play)
        while not self.core.is_current_trick_full:
            self.__simulate_next_opponent()

        trick, trick_winner_idx = self.core.complete_trick()
        trick_points = sum(points_for_card(c) for c in trick)

        # calculate the reward
        if self.reward_setting == 'sparse':
            reward = HeartsPlayEnvironment._calculate_sparse_reward(
                self.core.is_round_finished,
                self.core.current_round_points_collected
            )
        else:
            reward = HeartsPlayEnvironment._calculate_dense_reward(
                self.core.is_round_finished,
                self.core.current_round_points_collected,
                trick_points,
                trick_winner_idx == 0
            )

        is_round_finished = self.core.is_round_finished
        if not is_round_finished:
            while self.core.current_player_idx != 0:
                self.__simulate_next_opponent()

        return self._get_obs(), reward, is_round_finished, False, {}

    def action_masks(self) -> list[bool]:
        return create_play_env_action_masks_from_hearts_core(self.core)
