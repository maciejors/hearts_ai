from enum import Enum

import numpy as np


class Card:
    """
    Args:
        suit: Suit of the card
        rank_value: Numerical representation of card's rank (2-14, where Ace=14)
    """

    class Suit(Enum):
        CLUB = '\u2663'
        DIAMOND = '\u2666'
        SPADE = '\u2660'
        HEART = '\u2665'

    rank_mapper = {
        **{i: str(i) for i in range(2, 11)},
        11: 'J',
        12: 'Q',
        13: 'K',
        14: 'A',
    }

    def __init__(self, suit: Suit, rank_value: int):
        self.suit = suit
        self.rank = rank_value

    def __str__(self) -> str:
        return f'{Card.rank_mapper[self.rank]}{self.suit.value}'


class Deck:
    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

        # standard deck of 52 cards
        self._cards = np.array([
            Card(suit, rank)
            for suit in Card.Suit
            for rank in Card.rank_mapper.keys()
        ])
        self._cards_left = 0
        self.shuffle()

    def shuffle(self):
        """
        Resets and shuffles the deck
        """
        self._cards_left = len(self._cards)
        self._rng.shuffle(self._cards)

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
