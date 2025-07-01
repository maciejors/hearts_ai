from enum import Enum
from typing import Union

PLAYER_COUNT = 4
CARDS_IN_DECK_COUNT = 52
CARDS_PER_PLAYER_COUNT = CARDS_IN_DECK_COUNT // PLAYER_COUNT
CARDS_TO_PASS_COUNT = 3

HEART_POINTS = 1
Q_SPADES_POINTS = 13
MAX_POINTS = 26

Q_SPADES_IDX = 36
HEARTS_SUIT_IDX = 3
STARTING_CARD_IDX = 0


class PassDirection(Enum):
    LEFT = 0
    RIGHT = 1
    ACROSS = 2
    NO_PASSING = 3


class Suit(Enum):
    CLUB = '\u2663'
    DIAMOND = '\u2666'
    SPADE = '\u2660'
    HEART = '\u2665'

    @staticmethod
    def order(suit: Union['Suit', None]) -> int | None:
        if suit is None:
            return None
        return list(Suit).index(suit)
