import unittest

import numpy as np

from hearts_ai.game.hearts import HeartsGame
from hearts_ai.players import RandomPlayer


class TestHeartsGame(unittest.TestCase):
    def test_runs_with_random_players(self):
        # arrange
        players = [
            RandomPlayer(random_state=1),
            RandomPlayer(random_state=2),
            RandomPlayer(random_state=3),
            RandomPlayer(random_state=4),
        ]
        game = HeartsGame(
            players,
            random_state=5,
            rule_moon_shot=False,
        )
        # act
        game.play_round()
        # assert
        self.assertEqual(26, np.sum(game.scoreboard))
        self.assertEqual(26, np.sum(game.current_round_points))
        self.assertEqual([0] * 4, [len(hand) for hand in game.hands])
