from dataclasses import dataclass, field

import numpy as np

from hearts_ai.engine import Card, PassDirection, Suit
from hearts_ai.engine.constants import PLAYER_COUNT
from hearts_ai.rl.training.opponents.rule_based import play_card_rule_based, select_cards_to_pass_rule_based
from .base import BasePlayer
from ..deck import Deck


def _opponents_voids_default_factory() -> dict[Suit, list[bool]]:
    result = {}
    for suit in Suit:
        result[suit] = [False] * (PLAYER_COUNT - 1)
    return result


class RuleBasedPlayer(BasePlayer):
    @dataclass
    class Memory:
        # the suits the opponents have voided (True - voided, False - might have this suit)
        opponents_voids: dict[Suit, list[bool]] = field(default_factory=_opponents_voids_default_factory)
        remaining_cards_opponents: list[Card] = field(default_factory=lambda: list(Deck(123).all()))
        my_idx_in_curr_trick: int | None = None

    def __init__(self):
        self.memory = RuleBasedPlayer.Memory()

    def _get_indexes_of_players_in_trick(self) -> np.ndarray:
        """
        Returns:
            indexes of players in the list trick, relative to us
            (0 - us, 1 - left, 2 - across, 3 - right)
        """
        return (np.arange(PLAYER_COUNT) - self.memory.my_idx_in_curr_trick) % 4

    def play_card(
            self,
            hand: list[Card],
            trick: list[Card],
            are_hearts_broken: bool,
            is_first_trick: bool,
    ) -> Card:
        if is_first_trick:
            for card in hand:
                self.memory.remaining_cards_opponents.remove(card)

        self.memory.my_idx_in_curr_trick = len(trick)

        if len(trick) > 0:
            leading_suit = trick[0].suit
            for player_idx, card_played in zip(self._get_indexes_of_players_in_trick()[:len(trick)], trick):
                self.memory.remaining_cards_opponents.remove(card_played)
                if player_idx != 0 and card_played.suit != leading_suit:
                    self.memory.opponents_voids[leading_suit][player_idx - 1] = True

        return play_card_rule_based(
            hand=hand,
            current_trick=trick,
            are_hearts_broken=are_hearts_broken,
            is_first_trick=is_first_trick,
            opponents_voids=self.memory.opponents_voids,
            remaining_cards_opponents=self.memory.remaining_cards_opponents,
        )

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        return select_cards_to_pass_rule_based(
            hand=hand, direction=direction,
        )

    def post_trick_callback(self, trick: list[Card], is_trick_taken: bool) -> None:
        leading_suit = trick[0].suit
        for i, (player_idx, card_played) in enumerate(zip(self._get_indexes_of_players_in_trick(), trick)):
            if i <= self.memory.my_idx_in_curr_trick:
                # skip the part of trick already processed in play_card
                continue

            self.memory.remaining_cards_opponents.remove(card_played)
            if player_idx != 0 and card_played.suit != leading_suit:
                self.memory.opponents_voids[leading_suit][player_idx - 1] = True

    def post_round_callback(self, score: int) -> None:
        self.memory = RuleBasedPlayer.Memory()
