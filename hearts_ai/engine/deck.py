from dataclasses import dataclass
from typing import Self

import numpy as np

from .constants import Suit


@dataclass
class Card:
    """
    Args:
        suit: Suit of the card
        rank_value: Numerical representation of card's rank (2-14, where Ace=14)
    """
    suit: Suit
    rank_value: int

    rank_mapper = {
        **{i: str(i) for i in range(2, 11)},
        11: 'J',
        12: 'Q',
        13: 'K',
        14: 'A',
    }

    @property
    def rank(self) -> str:
        return Card.rank_mapper[self.rank_value]

    def __str__(self) -> str:
        return f'{Card.rank_mapper[self.rank_value]}{self.suit.value}'

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(str(self))


class Deck:
    """
    Standard 52 cards deck.
    Upon creating the object the deck is automatically shuffled
    """

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

        # standard deck of 52 cards
        self._cards = np.array([
            Card(suit, rank)
            for suit in Suit
            for rank in Card.rank_mapper.keys()
        ])
        self._cards_left = 0
        self.shuffle()

    def shuffle(self) -> Self:
        """
        Resets and shuffles the deck
        """
        self._cards_left = len(self._cards)
        self._rng.shuffle(self._cards)
        return self

    def deal(self, n: int) -> np.ndarray:
        """
        Deals n cards from the top
        """
        if n > self._cards_left:
            raise ValueError('Not enough cards left in the deck')

        start_idx = len(self._cards) - self._cards_left
        dealt_cards = self._cards[start_idx:start_idx + n]
        self._cards_left -= n

        return dealt_cards

    def all(self) -> np.ndarray:
        """Deal all remaining cards"""
        return self.deal(self._cards_left)
