import numpy as np

from .constants import HEART_POINTS, Q_SPADES_POINTS, HEARTS_SUIT_IDX, Q_SPADES_IDX


def is_heart(card_idx: int | np.int_) -> bool:
    return (card_idx // 13) == HEARTS_SUIT_IDX


def points_for_card(card_idx: int | np.int_) -> int:
    if is_heart(card_idx):
        return HEART_POINTS
    if card_idx == Q_SPADES_IDX:
        return Q_SPADES_POINTS
    return 0


def get_winning_card_argmax(cards: np.ndarray, leading_suit: int) -> np.int32:
    cards_matching_lead_suits = cards[cards // 13 == leading_suit]
    winning_card = np.max(cards_matching_lead_suits)
    return np.where(cards == winning_card)[0].item()


def get_valid_plays(hand: np.ndarray,
                    leading_suit: int | None,
                    are_hearts_broken: bool,
                    is_first_trick: bool) -> np.ndarray:
    """
    Returns:
        An array of valid card idx
    """
    if leading_suit is None and is_first_trick:
        # return a list with just the two of clubs = [0]
        return np.zeros(1, dtype=np.int16)

    if leading_suit is None:
        if not are_hearts_broken:
            non_hearts = hand[hand // 13 != HEARTS_SUIT_IDX]
            if len(non_hearts) > 0:
                return non_hearts
        return hand

    matching_suit = hand[hand // 13 == leading_suit]
    if len(matching_suit) > 0:
        return matching_suit

    if is_first_trick:
        non_points = hand[(hand // 13 != HEARTS_SUIT_IDX) & (hand != Q_SPADES_IDX)]
        if len(non_points) > 0:
            return non_points

    return hand
