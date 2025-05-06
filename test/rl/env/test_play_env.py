import unittest
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium.core import ObsType

from hearts_ai.engine import HeartsCore
from hearts_ai.rl.env import HeartsPlayEnvironment
from hearts_ai.rl.env.utils import card_to_idx
from test.utils import c, cl


def get_sample_env(reward_setting: Literal['dense', 'sparse', 'eval'] = 'dense',
                   card_passing_callbacks=None,
                   ) -> tuple[HeartsPlayEnvironment, ObsType]:
    env = HeartsPlayEnvironment(
        opponents_callbacks=MagicMock(),
        reward_setting=reward_setting,
        card_passing_callbacks=card_passing_callbacks,
    )
    obs_reset, _ = env.reset(seed=28)
    return env, obs_reset


def get_mock_params_allowing_step(agent_hand=None) -> dict:
    """
    Shared set of mocked attributes for HeartsCore,
    which allows to call .step(0) once with no errors or infinite loops
    """
    if agent_hand is None:
        agent_hand = [c('2♣')]
    return {
        'play_card': MagicMock(),
        'current_player_idx': 0,
        'is_current_trick_full': True,
        'hands': [agent_hand, [], [], []]
    }


def get_mock_complete_trick_14p(agent_takes=True) -> MagicMock:
    """Returns a sample trick with 14p"""
    trick_winner = 0 if agent_takes else 1
    return MagicMock(return_value=(cl(['3♦', '8♦', 'Q♠', '9♥']), trick_winner))


class TestHeartsPlayEnvironment(unittest.TestCase):

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        current_trick=cl(['2♣', '3♦', '10♣', 'Q♣']),
    )
    def test_trick_no(self):
        env, obs_reset = get_sample_env()
        obs_step, _, _, _, _ = env.step(0)

        self.assertEqual(1, obs_reset[0], 'Trick number should be 1 after reset()')
        self.assertEqual(2, obs_step[0], 'Trick number should increase after step()')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        is_round_finished=True,
    )
    def test_terminated_end_of_round(self):
        env, _ = get_sample_env()
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated, 'Environment should terminate at the end of round')

    @patch.multiple(
        HeartsCore,
        current_player_idx=0,
        hands=[cl(['2♣', '3♦']), [], [], []],
    )
    def test_agent_hand_start(self):
        env, obs = get_sample_env()

        card1_idx = card_to_idx(c('2♣'))
        card2_idx = card_to_idx(c('3♦'))
        self.assertEqual(1, obs[1 + card1_idx], 'Agent should have 2♣ in its hand')
        self.assertEqual(1, obs[1 + card2_idx], 'Agent should have 3♦ in its hand')
        self.assertEqual(2, sum(obs[1:53]), 'Agent should only have 2 cards in its hand')

    @patch.multiple(
        HeartsCore,
        current_player_idx=0,
        current_trick=cl(['3♦', 'A♣', 'Q♠', '9♥']),
    )
    def test_leading_suit_in_obs(self):
        env, obs = get_sample_env()
        self.assertEqual(1, obs[209 + 1], 'Diamonds should be the leading suit')
        self.assertEqual(1, sum(obs[209:213]), 'Only one suit should be marked as leading')

    @patch.multiple(
        HeartsCore,
        current_player_idx=0,
        current_round_points_collected=[1, 3, 3, 6],
    )
    def test_current_points_in_obs(self):
        env, obs = get_sample_env()
        self.assertListEqual([1, 3, 3, 6], list(obs[213:218]),
                             'Observation should contain current round points')

    @patch.multiple(
        HeartsCore,
        current_player_idx=0,
    )
    def test_invalid_action_no_change(self):
        env, obs_reset = get_sample_env()
        invalid_action = card_to_idx(c('3♦'))

        with patch('warnings.warn'):
            obs_invalid_action, reward, _, _, _ = env.step(invalid_action)

        np.testing.assert_array_equal(
            obs_reset, obs_invalid_action,
            'State should not change after an invalid action is taken'
        )
        self.assertEqual(0, reward, 'Reward should be 0 after an invalid action is taken')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
    )
    def test_dense_reward_agent_takes_trick(self):
        env, _ = get_sample_env('dense')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-14, reward,
                         'Agent should get a penalty after taking a trick '
                         'with points in the dense reward setting')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(agent_takes=False),
    )
    def test_dense_reward_agent_avoids_trick(self):
        env, _ = get_sample_env('dense')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(14, reward,
                         'Agent should get a reward after avoiding a trick '
                         'with points in the dense reward setting')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_points_collected=[14, 3, 3, 6],
    )
    def test_sparse_reward_zero_mid_game(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(0, reward, 'Sparse reward should be 0 mid-round')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_points_collected=[14, 3, 3, 6],
        is_round_finished=True,
    )
    def test_sparse_reward_regular_end_of_round(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-14, reward,
                         "Sparse reward should reflect agent's score at the end of round")

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_points_collected=[26, 0, 0, 0],
        is_round_finished=True,
    )
    def test_sparse_reward_agent_moon_shot(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(26, reward, 'Agent should get a reward after shooting the moon')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_points_collected=[0, 26, 0, 0],
        is_round_finished=True,
    )
    def test_sparse_reward_opponent_moon_shot(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-26, reward,
                         'Agent should get a penalty after an opponent shoots the moon')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_scores=[14, 3, 3, 6],
    )
    def test_eval_reward_zero_mid_game(self):
        env, _ = get_sample_env('eval')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(0, reward, 'Eval reward should be 0 mid-round')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        current_round_scores=[14, 3, 3, 6],
        is_round_finished=True,
    )
    def test_eval_reward_regular_end_of_round(self):
        env, _ = get_sample_env('eval')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-14, reward,
                         "Eval reward should reflect agent's score at the end of round")

    @patch.multiple(
        HeartsCore,
        **get_mock_params_allowing_step(agent_hand=[c('3♣'), c('5♠'), c('Q♥')]),
        current_trick=cl(['K♣']),
    )
    def test_action_mask_only_valid_cards(self):
        env, _ = get_sample_env()
        action_masks = env.action_masks()

        self.assertTrue(action_masks[card_to_idx(c('3♣'))])
        self.assertFalse(action_masks[card_to_idx(c('5♠'))])
        self.assertFalse(action_masks[card_to_idx(c('Q♥'))])

    @patch.multiple(
        HeartsCore,
        current_player_idx=0,
    )
    @patch('hearts_ai.engine.HeartsCore.complete_pass_cards')
    @patch('hearts_ai.engine.HeartsCore.pick_cards_to_pass')
    def test_card_passing(
            self,
            mock_pick_cards_to_pass: MagicMock,
            mock_complete_pass_cards: MagicMock,
    ):
        env, _ = get_sample_env(card_passing_callbacks=lambda *args: 0)
        mock_complete_pass_cards.assert_called_once()
        self.assertEqual(4, mock_pick_cards_to_pass.call_count)
