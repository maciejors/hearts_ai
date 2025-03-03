import numpy as np

from .deck import Deck
from players.base_player import BasePlayer


class HeartsGame:
    def __init__(self, players: list[BasePlayer], random_state: int | None = None):
        if len(players) != 4:
            raise ValueError('There should be exactly 4 players')

        self.players = players
        self.deck = Deck(random_state=random_state)
