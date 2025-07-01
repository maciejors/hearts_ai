import unittest
from typing import Literal
from unittest.mock import MagicMock, patch

import numpy as np
from gymnasium.core import ObsType

from hearts_ai.engine import HeartsRound
from hearts_ai.rl.env import HeartsPlayEnvironment
from test.utils import c, cla


def get_sample_env(reward_setting: Literal['dense', 'sparse', 'binary'] = 'dense',
                   card_passing_callbacks=None,
                   ) -> tuple[HeartsPlayEnvironment, ObsType]:
    env = HeartsPlayEnvironment(
        opponents_callbacks=MagicMock(),
        reward_setting=reward_setting,
        card_passing_callbacks=card_passing_callbacks,
    )
    obs_reset, _ = env.reset(seed=28)
    return env, obs_reset


def get_mock_numpy_state(put_at: int, relevant_part: np.ndarray) -> MagicMock:
    numpy_state = np.zeros(271, dtype=np.int8)
    numpy_state[put_at:put_at + len(relevant_part)] = relevant_part
    return MagicMock(side_effect=lambda: numpy_state.copy())


def get_mock_params_allowing_step(agent_hand=None) -> dict:
    """
    Shared set of mocked attributes for HeartsRound,
    which allows to call .step(0) once with no errors or infinite loops
    """
    if agent_hand is None:
        agent_hand = ['2♣']
    return {
        'play_card': MagicMock(),
        'current_player_idx': 0,
        'is_current_trick_full': True,
        'get_hand': MagicMock(return_value=cla(agent_hand)),
    }


def get_mock_complete_trick_14p(agent_takes=True) -> MagicMock:
    """Returns a sample trick with 14p"""
    trick_winner = 0 if agent_takes else 1
    return MagicMock(return_value=(cla(['3♦', '8♦', 'Q♠', '9♥']), trick_winner))


class TestHeartsPlayEnvironment(unittest.TestCase):

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        leading_suit=0,
        current_trick_unordered=cla(['2♣', '3♦', '10♣', 'Q♣']),
    )
    def test_trick_no(self):
        env, obs_reset = get_sample_env()
        obs_step, _, _, _, _ = env.step(0)

        self.assertEqual(1, obs_reset[0], 'Trick number should be 1 after reset()')
        self.assertEqual(2, obs_step[0], 'Trick number should increase after step()')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        is_finished=True,
    )
    def test_terminated_end_of_round(self):
        env, _ = get_sample_env()
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated, 'Environment should terminate at the end of round')

    @patch.multiple(
        HeartsRound,
        get_numpy_state=get_mock_numpy_state(1, np.array([1, 0, 1]))
    )
    def test_agent_hand_start(self):
        _, obs = get_sample_env()

        card1_idx = c('2♣').idx
        card2_idx = c('4♣').idx
        self.assertEqual(1, obs[1 + card1_idx], 'Agent should have 2♣ in its hand')
        self.assertEqual(1, obs[1 + card2_idx], 'Agent should have 3♦ in its hand')
        self.assertEqual(2, sum(obs[1:53]), 'Agent should only have 2 cards in its hand')

    @patch.multiple(
        HeartsRound,
        get_numpy_state=get_mock_numpy_state(209, np.array([0, 1, 0, 0]))
    )
    def test_leading_suit_in_obs(self):
        _, obs = get_sample_env()
        self.assertEqual(1, obs[209 + 1], 'Diamonds should be the leading suit')
        self.assertEqual(1, sum(obs[209:213]), 'Only one suit should be marked as leading')

    @patch.multiple(
        HeartsRound,
        get_numpy_state=get_mock_numpy_state(213, np.array([1, 3, 3, 6])),
    )
    def test_current_points_in_obs(self):
        _, obs = get_sample_env()
        np.testing.assert_array_equal(obs[213:218], np.array([1, 3, 3, 6]),
                             'Observation should contain current round points')

    @patch.multiple(
        HeartsRound,
        current_player_idx=0,
    )
    def test_invalid_action_no_change(self):
        env, obs_reset = get_sample_env()
        invalid_action = c('3♦').idx

        with patch('warnings.warn'):
            obs_invalid_action, reward, _, _, _ = env.step(invalid_action)

        np.testing.assert_array_equal(
            obs_reset, obs_invalid_action,
            'State should not change after an invalid action is taken'
        )
        self.assertEqual(0, reward, 'Reward should be 0 after an invalid action is taken')

    @patch.multiple(
        HeartsRound,
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
        HeartsRound,
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
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        points_collected=np.array([14, 3, 3, 6]),
    )
    def test_sparse_reward_zero_mid_game(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(0, reward, 'Sparse reward should be 0 mid-round')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        points_collected=np.array([14, 3, 3, 6]),
        is_finished=True,
    )
    def test_sparse_reward_regular_end_of_round(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-14, reward,
                         "Sparse reward should reflect agent's score at the end of round")

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        points_collected=np.array([26, 0, 0, 0]),
        is_finished=True,
    )
    def test_sparse_reward_agent_moon_shot(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(26, reward, 'Agent should get a reward after shooting the moon')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        points_collected=np.array([0, 26, 0, 0]),
        is_finished=True,
    )
    def test_sparse_reward_opponent_moon_shot(self):
        env, _ = get_sample_env('sparse')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(-26, reward,
                         'Agent should get a penalty after an opponent shoots the moon')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        scores=np.array([0, 14, 3, 6]),
    )
    def test_binary_reward_zero_mid_game(self):
        env, _ = get_sample_env('binary')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(0, reward, 'Binary reward should be 0 mid-round')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        scores=np.array([14, 3, 3, 6]),
        is_finished=True,
    )
    def test_binary_reward_zero_loss(self):
        env, _ = get_sample_env('binary')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(0, reward,
                         'Binary reward should be 0 when agent is not the winner of a round')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        scores=np.array([3, 3, 14, 6]),
        is_finished=True,
    )
    def test_binary_reward_one_win(self):
        env, _ = get_sample_env('binary')
        _, reward, _, _, _ = env.step(0)
        self.assertEqual(1, reward,
                         'Binary reward should be 1 when agent has the lowest score')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        scores=np.array([14, 3, 3, 6]),
        is_finished=True,
    )
    def test_is_success_loss(self):
        env, _ = get_sample_env()
        _, _, _, _, info = env.step(0)
        self.assertFalse(info['is_success'],
                         'is_success should be False when agent is not the winner of a round')

    @patch.multiple(
        HeartsRound,
        **get_mock_params_allowing_step(),
        complete_trick=get_mock_complete_trick_14p(),
        scores=np.array([3, 3, 14, 6]),
        is_finished=True,
    )
    def test_is_success_win(self):
        env, _ = get_sample_env()
        _, _, _, _, info = env.step(0)
        self.assertTrue(info['is_success'],
                        'is_success should be True when agent has the lowest score')

    @patch.multiple(
        HeartsRound,
        get_hand=MagicMock(return_value=cla(['3♣', '5♠', 'Q♥'])),
        leading_suit=0,
        are_hearts_broken=False,
        trick_no=2,
    )
    def test_action_mask_only_valid_cards(self):
        env, _ = get_sample_env()
        action_masks = env.action_masks()

        self.assertTrue(action_masks[c('3♣').idx])
        self.assertFalse(action_masks[c('5♠').idx])
        self.assertFalse(action_masks[c('Q♥').idx])

    @patch.multiple(
        HeartsRound,
        current_player_idx=0,
    )
    @patch('hearts_ai.engine.HeartsRound.perform_cards_passing')
    @patch('hearts_ai.engine.HeartsRound.pick_cards_to_pass')
    def test_card_passing(
            self,
            mock_pick_cards_to_pass: MagicMock,
            mock_complete_pass_cards: MagicMock,
    ):
        get_sample_env(card_passing_callbacks=lambda *args: 0)
        mock_complete_pass_cards.assert_called_once()
        self.assertEqual(4, mock_pick_cards_to_pass.call_count)
