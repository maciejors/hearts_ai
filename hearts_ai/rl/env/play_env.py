import warnings
from typing import Literal, Any, SupportsFloat, Callable

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.engine import HeartsCore, HeartsRules, Suit, Card
from hearts_ai.engine.utils import points_for_card, get_valid_plays
from .utils import card_to_idx, action_to_card


def create_play_env_obs(
        trick_no: int,
        player_idx: int,
        trick_starting_player_idx: int,
        current_trick: list[Card],
        hands: list[list[Card]],
        played_cards: list[list[Card]],
        current_round_points_collected: list[int],
) -> ObsType:
    """
    Returns the current state from the perspective of the current player.

    For details on the observation space refer to :class:`HeartsPlayEnvironment`
    """
    state = np.zeros(217, dtype=np.int8)

    state[0] = trick_no
    current_trick = current_trick

    for game_player_idx, (hand, played_cards) in enumerate(zip(hands, played_cards)):
        # player index within the state. This is to account for the fact that
        # each state looks differently from each player's perspective
        state_player_idx = (game_player_idx - player_idx) % 4
        # hands
        cards_in_hand_idx = np.array([
            card_to_idx(card) for card in hand
        ], dtype=np.int16)
        state[cards_in_hand_idx + 1 + 52 * state_player_idx] = 1

        # played
        cards_played_idx = np.array([
            card_to_idx(card) for card in played_cards
        ], dtype=np.int16)
        state[cards_played_idx + 1 + 52 * state_player_idx] = -1

        # current trick
        player_idx_in_trick = (game_player_idx - trick_starting_player_idx) % 4
        if player_idx_in_trick < len(current_trick):
            player_card_in_trick_idx = card_to_idx(
                current_trick[player_idx_in_trick]
            )
            state[player_card_in_trick_idx + 1 + 52 * state_player_idx] = 2

    # leading suit
    if len(current_trick) > 0:
        leading_suit_state_idx = list(Suit).index(current_trick[0].suit) + 209
        state[leading_suit_state_idx] = 1

    # round points
    state[213:217] = np.array(current_round_points_collected).astype(np.int8)
    return state


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
            returning an action
        reward_setting: Whether to use the sparse or dense reward setting.
            Default: dense
    """

    def __init__(self,
                 opponents_callbacks: list[Callable[[ObsType, list[bool]], ActType]],
                 reward_setting: Literal['dense', 'sparse'] = 'dense'):
        super().__init__()

        self.opponents_callbacks = opponents_callbacks

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
        return create_play_env_obs(
            trick_no=self.core.trick_no,
            player_idx=self.core.current_player_idx,
            trick_starting_player_idx=self.core.trick_starting_player_idx,
            current_trick=self.core.current_trick,
            hands=self.core.hands,
            played_cards=self.core.played_cards,
            current_round_points_collected=self.core.current_round_points_collected,
        )

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
        hand = self.core.hands[self.core.current_player_idx]
        valid_plays = get_valid_plays(
            hand=hand,
            trick=self.core.current_trick,
            are_hearts_broken=self.core.are_hearts_broken,
            is_first_trick=self.core.trick_no == 1,
        )
        valid_plays_indices = [card_to_idx(c) for c in valid_plays]

        mask_np = np.full(52, False)
        mask_np[valid_plays_indices] = True
        return mask_np.tolist()
