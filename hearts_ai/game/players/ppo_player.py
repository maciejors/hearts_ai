from academia.agents import PPOAgent

from hearts_ai.engine import Card, PassDirection
from .base import BasePlayer


class PPOPlayer(BasePlayer):

    def __init__(self, playing_model: PPOAgent):
        self.playing_model = playing_model

    def play_card(self, hand: list[Card], trick: list[Card], are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        pass

    def select_cards_to_pass(self, hand: list[Card], direction: PassDirection) -> list[Card]:
        raise NotImplementedError('For now, this agent does not support the card pass feature')

    def __form_state(self, hand: list[Card], trick: list[Card]):
        """Form the curent state using the known information as well as determinisation techniques"""
