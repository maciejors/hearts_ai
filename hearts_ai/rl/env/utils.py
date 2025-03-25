from gymnasium.core import ActType

from hearts_ai.engine import Card, Suit


def action_to_card(action: ActType) -> Card:
    suits_ordered = list(Suit)
    suit = suits_ordered[action // 13]
    rank_value = action % 13 + 2
    return Card(suit, rank_value)


def card_to_idx(card: Card):
    suit_idx = list(Suit).index(card.suit)
    rank_idx = card.rank_value - 2
    return suit_idx * 13 + rank_idx
