from .constants import Suit, HEART_POINTS, Q_SPADES_POINTS
from .deck import Card


def is_q_spades(card: Card) -> bool:
    return card.suit == Suit.SPADE and card.rank == 'Q'


def is_heart(card: Card) -> bool:
    return card.suit == Suit.HEART


def is_starting_card(card: Card) -> bool:
    return card.suit == Suit.CLUB and card.rank == '2'


def points_for_card(card: Card) -> int:
    if is_heart(card):
        return HEART_POINTS
    if is_q_spades(card):
        return Q_SPADES_POINTS
    return 0
