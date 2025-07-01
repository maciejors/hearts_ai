import numpy as np
from gymnasium.core import ObsType

from hearts_ai.engine import HeartsRound, PassDirection
from hearts_ai.engine.round import STATE_IDX_HANDS_INFO_START, STATE_IDX_POINTS_COLLECTED
from hearts_ai.engine.utils import get_valid_plays


def create_play_env_obs(
        trick_no: int,
        player_idx: int,
        trick_starting_player_idx: int,
        current_trick_ordered: np.ndarray,
        hands: list[np.ndarray],
        played_cards: list[np.ndarray],
        current_round_points_collected: np.ndarray,
) -> ObsType:
    """
    Returns the current state from the perspective of the current player.

    For details on the observation space refer to :class:`HeartsPlayEnvironment`
    """
    state = np.zeros(217, dtype=np.int8)
    state[0] = trick_no

    for game_player_idx, (hand, played_cards) in enumerate(zip(hands, played_cards)):
        # player index within the state. This is to account for the fact that
        # each state looks differently from each player's perspective
        state_player_idx = (game_player_idx - player_idx) % 4

        state[hand + 1 + 52 * state_player_idx] = 1
        state[played_cards + 1 + 52 * state_player_idx] = -1

        # current trick
        player_idx_in_trick = (game_player_idx - trick_starting_player_idx) % 4
        if player_idx_in_trick < len(current_trick_ordered):
            player_card_in_trick_idx = current_trick_ordered[player_idx_in_trick]
            state[player_card_in_trick_idx + 1 + 52 * state_player_idx] = 2

    # leading suit
    if len(current_trick_ordered) > 0:
        leading_suit = current_trick_ordered[0] // 13
        state[209 + leading_suit] = 1

    # round points
    state[213:217] = current_round_points_collected
    return state


def create_play_env_obs_from_hearts_round(hearts_round: HeartsRound) -> ObsType:
    obs = hearts_round.get_numpy_state()[:217]
    current_player_idx = int(hearts_round.current_player_idx)

    hands_idx = np.array([
        np.arange(idx_start, idx_start + 52)
        for idx_start in STATE_IDX_HANDS_INFO_START
    ])
    points_idx = np.array(STATE_IDX_POINTS_COLLECTED)

    # we want the current player to be at index 0 in the state
    hands_idx_shifted = np.roll(hands_idx, shift=-current_player_idx, axis=0)
    points_idx_shifted = np.roll(points_idx, shift=-current_player_idx)

    obs[hands_idx.flatten()] = obs[hands_idx_shifted.flatten()]
    obs[points_idx] = obs[points_idx_shifted]
    return obs


def create_play_env_action_masks(
        hand: np.ndarray,
        leading_suit: int | None,
        are_hearts_broken: bool,
        is_first_trick: bool,
) -> np.ndarray:
    """
    Returns the action masks from the perspective of the current player.

    For details on the action space refer to :class:`HeartsPlayEnvironment`
    """
    valid_plays = get_valid_plays(
        hand=hand,
        leading_suit=leading_suit,
        are_hearts_broken=are_hearts_broken,
        is_first_trick=is_first_trick,
    )
    mask = np.full(52, False)
    mask[valid_plays] = True
    return mask


def create_play_env_action_masks_from_hearts_round(hearts_round: HeartsRound) -> np.ndarray:
    return create_play_env_action_masks(
        hand=hearts_round.get_hand(hearts_round.current_player_idx),
        leading_suit=hearts_round.leading_suit,
        are_hearts_broken=hearts_round.are_hearts_broken,
        is_first_trick=hearts_round.trick_no == 1,
    )


def create_cards_pass_env_obs(
        player_hand: np.ndarray,
        picked_cards: np.ndarray,
        pass_direction: PassDirection,
) -> ObsType:
    """
    For details on the observation space refer to :class:`HeartsCardPassEnvironment`
    """
    state = np.zeros(55, dtype=np.int8)
    state[player_hand] = 1
    state[picked_cards] = -1
    state[52 + pass_direction.value] = 1
    return state


def create_cards_pass_env_action_masks(
        player_hand: np.ndarray,
        picked_cards: np.ndarray,
) -> np.ndarray:
    valid_cards = np.setdiff1d(player_hand, picked_cards)
    mask = np.full(52, False)
    mask[valid_cards] = True
    return mask
