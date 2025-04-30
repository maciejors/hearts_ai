from collections.abc import Iterable
from typing import TypeVar, Callable

from gymnasium.core import ActType

from hearts_ai.engine import Card, Suit

_ActTypeGeneric = TypeVar('_ActTypeGeneric')
_ObsTypeGeneric = TypeVar('_ObsTypeGeneric')
ActionTakingCallback = Callable[[_ObsTypeGeneric, list[bool]], _ActTypeGeneric]
ActionTakingCallbackParam = ActionTakingCallback | list[ActionTakingCallback]


def handle_action_taking_callback_param(p: ActionTakingCallbackParam,
                                        list_len: int,
                                        ) -> list[ActionTakingCallback]:
    if isinstance(p, Iterable):
        return p
    return [p] * list_len


def action_to_card(action: ActType) -> Card:
    suits_ordered = list(Suit)
    suit = suits_ordered[action // 13]
    rank_value = action % 13 + 2
    return Card(suit, rank_value)


def card_to_idx(card: Card):
    suit_idx = list(Suit).index(card.suit)
    rank_idx = card.rank_value - 2
    return suit_idx * 13 + rank_idx
