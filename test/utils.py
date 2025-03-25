"""
Utilities for tests
"""
from hearts_ai.engine import Card, Suit


def c(card_str: str) -> Card:
    """
    A quick way to parse a Card object from a string e.g. "10♥"
    """
    suit_str = card_str[-1]
    suit = [s for s in list(Suit) if s.value == suit_str][0]

    rank_str = card_str[:-1]
    reverse_rank_mapper = {v: k for k, v in Card.rank_mapper.items()}
    rank_value = reverse_rank_mapper[rank_str]

    return Card(suit, rank_value)


def cl(cards_str: list[str]) -> list[Card]:
    """
    A quick way to parse a list of Card object from a string e.g. ["10♥", "Q♣"]
    """
    return [c(s) for s in cards_str]
