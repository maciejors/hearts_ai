import numpy as np
from gymnasium.core import ObsType

from hearts_ai.engine import HeartsRound, Suit, Card, PassDirection
from hearts_ai.engine.utils import get_valid_plays
from hearts_ai.rl.env.utils import card_to_idx


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


def create_play_env_obs_from_hearts_round(hearts_round: HeartsRound) -> ObsType:
    return create_play_env_obs(
        trick_no=hearts_round.trick_no,
        player_idx=hearts_round.current_player_idx,
        trick_starting_player_idx=hearts_round.trick_starting_player_idx,
        current_trick=hearts_round.current_trick,
        hands=hearts_round.hands,
        played_cards=hearts_round.played_cards,
        current_round_points_collected=hearts_round.points_collected,
    )


def create_play_env_action_masks(
        hand: list[Card],
        current_trick: list[Card],
        are_hearts_broken: bool,
        is_first_trick: bool,
) -> np.ndarray:
    """
    Returns the action masks from the perspective of the current player.

    For details on the action space refer to :class:`HeartsPlayEnvironment`
    """
    valid_plays = get_valid_plays(
        hand=hand,
        trick=current_trick,
        are_hearts_broken=are_hearts_broken,
        is_first_trick=is_first_trick,
    )
    valid_plays_indices = [card_to_idx(c) for c in valid_plays]

    mask = np.full(52, False)
    mask[valid_plays_indices] = True
    return mask


def create_play_env_action_masks_from_hearts_round(hearts_round: HeartsRound) -> np.ndarray:
    return create_play_env_action_masks(
        hand=hearts_round.hands[hearts_round.current_player_idx],
        current_trick=hearts_round.current_trick,
        are_hearts_broken=hearts_round.are_hearts_broken,
        is_first_trick=hearts_round.trick_no == 1,
    )


def create_cards_pass_env_obs(
        player_hand: list[Card],
        picked_cards: list[Card],
        pass_direction: PassDirection,
) -> ObsType:
    """
    For details on the observation space refer to :class:`HeartsCardPassEnvironment`
    """
    state = np.zeros(55, dtype=np.int8)

    cards_in_hand_idx = np.array([
        card_to_idx(card) for card in player_hand
    ], dtype=np.int16)
    state[cards_in_hand_idx] = 1

    picked_cards_idx = np.array([
        card_to_idx(card) for card in picked_cards
    ], dtype=np.int16)
    state[picked_cards_idx] = -1

    state[52 + pass_direction.value] = 1
    return state


def create_cards_pass_env_action_masks(
        player_hand: list[Card],
        picked_cards: list[Card],
        pass_direction: PassDirection,
) -> np.ndarray:
    obs = create_cards_pass_env_obs(player_hand, picked_cards, pass_direction)
    return np.array(obs[:52] == 1)
