import unittest
from unittest.mock import patch

import numpy as np

from hearts_ai.engine import HeartsRules, PassDirection
from hearts_ai.game import HeartsGame
from hearts_ai.game.players import RandomPlayer, RuleBasedPlayer


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
        self.assertEqual(26, np.sum(game.core.current_round_scores),
                         'Sum of all points should be equal to 26')
        self.assertEqual([0] * 4, [len(hand) for hand in game.core._hands],
                         'All players should have empty hands')
        self.assertEqual([13] * 4, [len(cards) for cards in game.core._played_cards],
                         'All players should have played 13 cards')
        self.assertEqual(
            52,
            np.sum([len(cards_for_player) for cards_for_player in game.core._taken_cards]),
            'There should be 52 cards across all tricks taken',
        )

    def test_rule_based_players_do_not_raise_errors(self):
        players = [
            RuleBasedPlayer() for _ in range(4)
        ]
        game = HeartsGame(
            players,
            rules=HeartsRules(),
            random_state=5,
        )
        for _ in range(10):
            game.play_round()

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

    def test_post_trick_callbacks_called(self):
        with patch.object(RandomPlayer, 'post_trick_callback') as mocked_callback:
            # arrange
            game = get_game_with_random_players()
            # act
            game.play_round()
            # assert
            self.assertEqual(4 * 13, len(mocked_callback.mock_calls))

    def test_post_round_callbacks_called(self):
        with patch.object(RandomPlayer, 'post_round_callback') as mocked_callback:
            # arrange
            game = get_game_with_random_players()
            # act
            game.play_round()
            # assert
            self.assertEqual(4, len(mocked_callback.mock_calls))
