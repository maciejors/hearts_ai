import numpy as np

from hearts_ai.engine.card import Card
from hearts_ai.engine.constants import PassDirection
from .base.base_player import BasePlayer
from ..utils import get_valid_plays_objs


class RandomPlayer(BasePlayer):

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

    def play_card(self,
                  hand: list[Card],
                  trick: list[Card],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        leading_suit = trick[0].suit if len(trick) > 0 else None
        valid_plays = get_valid_plays_objs(hand, leading_suit, are_hearts_broken, is_first_trick)
        chosen_card_idx = self._rng.integers(len(valid_plays))
        return valid_plays[chosen_card_idx]

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        chosen_cards_idx = self._rng.choice(range(len(hand)), size=3, replace=False)
        return [hand[i] for i in chosen_cards_idx]
