from .constants import Suit


class Card:
    """
    Args:
        idx: Value from the range 0-51 representing the index of the card.
            The cards are ordered by suit: clubs, diamonds, spades, hearts;
            and within each suit by rank: from 2 to Ace
    """

    def __init__(self, idx: int):
        self.idx = idx

    ranks_str = ('2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A')

    @classmethod
    def of(cls, rank: str, suit: Suit):
        suit_order = list(Suit).index(suit)
        rank_order = Card.ranks_str.index(rank)
        return cls(suit_order * 13 + rank_order)

    @property
    def rank(self) -> str:
        return Card.ranks_str[self.idx % 13]

    @property
    def rank_value(self) -> int:
        """
        Returns:
            Value from 2 to 14, where 14 = Ace
        """
        return self.idx % 13 + 2

    @property
    def suit(self) -> Suit:
        return list(Suit)[self.idx // 13]

    def __str__(self) -> str:
        return f'{self.rank}{self.suit.value}'

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self):
        return hash(self.idx)

    def __index__(self):
        return self.idx

    def __eq__(self, other):
        if type(other) != Card:
            return False
        return self.idx == other.idx
