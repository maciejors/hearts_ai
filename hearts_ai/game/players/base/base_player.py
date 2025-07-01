from abc import ABC, abstractmethod

from hearts_ai.engine.card import Card
from hearts_ai.engine.constants import PassDirection


class BasePlayer(ABC):
    """Abstract base class for all Hearts players"""

    @abstractmethod
    def play_card(self,
                  hand: list[Card],
                  trick: list[Card],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        raise NotImplementedError()

    @abstractmethod
    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        raise NotImplementedError()

    def post_trick_callback(self, trick: list[Card], is_trick_taken: bool) -> None:
        """A method which is called after every trick to inform a player about its outcome"""
        pass

    def post_round_callback(self, score: int) -> None:
        """A method which is called after every round to inform a player about their score"""
        pass
