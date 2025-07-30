import numpy as np

from hearts_ai.engine import Card, Suit
from hearts_ai.engine.utils import get_valid_plays


def card_list_to_array(card_list: list[Card]) -> np.ndarray:
    return np.array([c.idx for c in card_list], dtype=np.int16)


def array_to_card_list(arr: np.ndarray) -> list[Card]:
    return [Card(i) for i in arr]


def get_valid_plays_objs(
        hand: list[Card],
        leading_suit: Suit | None,
        are_hearts_broken: bool,
        is_first_trick: bool,
) -> list[Card]:
    valid_plays_arr = get_valid_plays(
        hand=card_list_to_array(hand),
        leading_suit=Suit.order(leading_suit),
        are_hearts_broken=are_hearts_broken,
        is_first_trick=is_first_trick,
    )
    return array_to_card_list(valid_plays_arr)
