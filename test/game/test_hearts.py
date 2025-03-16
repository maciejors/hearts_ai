import unittest
from unittest import mock

import numpy as np

from hearts_ai.game.hearts import HeartsGame
from hearts_ai.players import RandomPlayer


def get_game_with_random_players() -> HeartsGame:
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
    return game


class TestHeartsGame(unittest.TestCase):
    def test_play_round(self):
        # arrange
        game = get_game_with_random_players()
        # act
        game.play_round()
        # assert
        self.assertEqual(26, np.sum(game.current_round_points),
                         'Sum of all points should be equal to 26')
        self.assertEqual(26, np.sum(game.scoreboard),
                         'Scoreboard should be updated after pre_round() is called')
        self.assertEqual([0] * 4, [len(hand) for hand in game.hands],
                         'All players should have empty hands')
        self.assertEqual(
            52,
            np.sum([len(cards_for_player) for cards_for_player in game.taken_cards]),
            'There should be 52 cards across all tricks taken',
        )

    def test_post_trick_callbacks_called(self):
        with mock.patch.object(RandomPlayer, 'post_trick_callback') as mocked_callback:
            # arrange
            game = get_game_with_random_players()
            # act
            game.pre_round()
            game.play_trick()
            # assert
            self.assertEqual(4, len(mocked_callback.mock_calls))

    def test_post_round_callbacks_called(self):
        with mock.patch.object(RandomPlayer, 'post_round_callback') as mocked_callback:
            # arrange
            game = get_game_with_random_players()
            # act
            game.play_round()
            # assert
            self.assertEqual(4, len(mocked_callback.mock_calls))
