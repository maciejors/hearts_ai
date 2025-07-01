"""
Utilities for tests
"""
import numpy as np

from hearts_ai.engine import Card, Suit


def c(card_str: str) -> Card:
    """
    A quick way to parse a Card object from a string e.g. "10♥"
    """
    rank_str = card_str[:-1]
    suit_str = card_str[-1]
    suit = [s for s in list(Suit) if s.value == suit_str][0]
    return Card.of(rank_str, suit)


def cl(cards_str: list[str]) -> list[Card]:
    """
    A quick way to parse a list of Card object from a string e.g. ["10♥", "Q♣"]
    """
    return [c(s) for s in cards_str]


def cla(cards_str: list[str]) -> np.array:
    """
    Same as ``cl`` but converts to an array (useful when mocking core and envs)
    """
    return np.array([card.idx for card in cl(cards_str)])
