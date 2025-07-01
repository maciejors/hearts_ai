from typing import Self

import numpy as np

from hearts_ai.engine import Card


class Deck:
    """
    Standard 52 cards deck.
    Upon creating the object the deck is automatically shuffled
    """

    __slots__ = ['_rng', '_cards', '_cards_left']

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

        # standard deck of 52 cards
        self._cards = np.array([
            Card(i) for i in range(52)
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
