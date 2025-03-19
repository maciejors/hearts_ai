import unittest

import numpy as np

from hearts_ai.engine import HeartsRules, PassDirection
from hearts_ai.game import HeartsGame
from hearts_ai.game.players import RandomPlayer


def get_game_with_random_players() -> HeartsGame:
    players = [
        RandomPlayer(random_state=1),
        RandomPlayer(random_state=2),
        RandomPlayer(random_state=3),
        RandomPlayer(random_state=4),
    ]
    game = HeartsGame(
        players,
        rules=HeartsRules(moon_shot=False),
        random_state=5,
    )
    return game


class TestHeartsGame(unittest.TestCase):
    def test_play_round(self):
        # arrange
        game = get_game_with_random_players()
        # act
        game.play_round()
        # assert
        self.assertEqual(26, np.sum(game.core.current_round_points),
                         'Sum of all points should be equal to 26')
        self.assertEqual([0] * 4, [len(hand) for hand in game.core.hands],
                         'All players should have empty hands')
        self.assertEqual(
            52,
            np.sum([len(cards_for_player) for cards_for_player in game.core.taken_cards]),
            'There should be 52 cards across all tricks taken',
        )

    def test_complete_round(self):
        # arrange
        game = get_game_with_random_players()
        # act
        game.play_round()
        game.play_round()
        # assert
        self.assertEqual(26, np.sum(game.core.scoreboard),
                         'Scoreboard should be updated after the next round starts')

    def test_passing_cards_changes(self):
        game = get_game_with_random_players()

        game.play_round()
        self.assertEqual(PassDirection.LEFT, game.core.pass_direction)

        game.play_round()
        self.assertEqual(PassDirection.RIGHT, game.core.pass_direction)

        game.play_round()
        self.assertEqual(PassDirection.ACROSS, game.core.pass_direction)

        game.play_round()
        self.assertEqual(PassDirection.NO_PASSING, game.core.pass_direction)

        game.play_round()
        self.assertEqual(PassDirection.LEFT, game.core.pass_direction)
