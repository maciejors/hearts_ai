from enum import Enum

PLAYER_COUNT = 4
CARDS_IN_DECK = 52
CARDS_PER_PLAYER = CARDS_IN_DECK // PLAYER_COUNT

HEART_POINTS = 1
Q_SPADES_POINTS = 13
MAX_POINTS = 26


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
