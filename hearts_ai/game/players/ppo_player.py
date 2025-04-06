from sb3_contrib import MaskablePPO

from hearts_ai.engine import Card, PassDirection
from .base import BasePlayer


class PPOPlayer(BasePlayer):

    def __init__(self, playing_agent: MaskablePPO, card_passing_agent: MaskablePPO):
        self.playing_agent = playing_agent
        self.card_passing_agent = card_passing_agent

    def play_card(self,
                  hand: list[Card],
                  trick: list[Card],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        pass

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        pass

    def __determinisation_state(self, hand: list[Card], trick: list[Card]):
        """Form the curent state using the known information as well as determinisation techniques"""
        pass
