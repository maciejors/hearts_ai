import numpy as np

from constants import PassDirection
from hearts_ai.game.deck import Card
from hearts_ai.players.base.base_player import BasePlayer


class RandomPlayer(BasePlayer):

    def __init__(self, random_state: int | None = None):
        self._rng = np.random.default_rng(random_state)

    def play_card(self,
                  hand: list[Card],
                  trick: list[Card],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        valid_plays = self._get_valid_plays(hand, trick, are_hearts_broken, is_first_trick)
        chosen_card_idx = self._rng.integers(len(valid_plays))
        return valid_plays[chosen_card_idx]

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        chosen_cards_idx = self._rng.choice(range(len(hand)), size=3, replace=False)
        return [hand[i] for i in chosen_cards_idx]
