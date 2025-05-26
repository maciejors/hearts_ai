import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
from gymnasium.core import ObsType

from hearts_ai.engine import HeartsCore
from hearts_ai.rl.env import HeartsCardsPassEnvironment
from hearts_ai.rl.env.utils import card_to_idx
from test.utils import c, cl


def get_sample_env() -> tuple[HeartsCardsPassEnvironment, ObsType]:
    env = HeartsCardsPassEnvironment(
        opponents_callbacks=MagicMock(),
        playing_callbacks=MagicMock(),
    )
    obs_reset, _ = env.reset(seed=28)
    return env, obs_reset


def get_mock_params_for_hearts_core() -> dict:
    """
    Shared set of mocked attributes for HeartsCore
    """
    # looks quirky but will ensure that scores are different every time,
    # so the reward should never be 0 if correctly calculated
    i = 0

    def next_val():
        nonlocal i
        i += 1
        return i

    return {
        'pick_cards_to_pass': MagicMock(),
        'complete_pass_cards': MagicMock(),
        'is_round_finished': True,
        'current_round_scores': PropertyMock(side_effect=lambda: [next_val(), 0, 0, 0]),
    }


class TestHeartsPlayEnvironment(unittest.TestCase):

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣', '3♦']), [], [], []],
    )
    def test_reset_state(self):
        env, obs_reset = get_sample_env()
        card1_idx = card_to_idx(c('2♣'))
        card2_idx = card_to_idx(c('3♦'))
        self.assertEqual(1, obs_reset[card1_idx], 'Agent should have 2♣ in its hand')
        self.assertEqual(1, obs_reset[card2_idx], 'Agent should have 3♦ in its hand')
        self.assertEqual(2, np.sum(obs_reset[:52]), 'Agent should only have 2 cards in its hand')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    def test_step_state(self):
        with patch.object(HeartsCardsPassEnvironment.State, 'hands', [cl(['2♣', '3♦', '4♦']), [], [], []]):
            env, _ = get_sample_env()
            card1_idx = card_to_idx(c('2♣'))
            card2_idx = card_to_idx(c('3♦'))
            card3_idx = card_to_idx(c('4♦'))

            obs = env.step(card3_idx)[0]

            self.assertEqual(1, obs[card1_idx], 'Agent should have 2♣ in its hand')
            self.assertEqual(1, obs[card2_idx], 'Agent should have 3♦ in its hand')
            self.assertEqual(-1, obs[card3_idx], 'Agent should have 4♦ as picked')
            self.assertEqual(52 - 3, np.sum(obs[:52] == 0),
                             'All other cards should have a value of 0 in the state')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    def test_state_pass_direction(self):
        env, obs_reset_init = get_sample_env()

        obs_reset_all = [obs_reset_init]
        for i in range(3):
            obs_reset_all.append(env.reset(seed=i)[0])

        self.assertEqual(1, obs_reset_all[0][52],
                         'Pass direction should be left after the first reset')
        self.assertEqual(1, obs_reset_all[1][53],
                         'Pass direction should be right after the second reset')
        self.assertEqual(1, obs_reset_all[2][54],
                         'Pass direction should be across after the third reset')
        self.assertEqual(1, obs_reset_all[3][52],
                         'Pass direction should be left after the fourth reset')
        for i, obs_reset in enumerate(obs_reset_all):
            self.assertEqual(1, np.sum(obs_reset[52:55]), 'Pass direction should be one-hot encoded')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣', '3♦']), [], [], []],
    )
    def test_action_mask_hand(self):
        env, _ = get_sample_env()
        card1_idx = card_to_idx(c('2♣'))
        card2_idx = card_to_idx(c('3♦'))
        action_mask_reset = env.action_masks()
        self.assertTrue(action_mask_reset[card1_idx])
        self.assertTrue(action_mask_reset[card2_idx])
        self.assertEqual(2, np.sum(action_mask_reset), 'There should only be 2 valid plays')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣', '3♦']), [], [], []],
    )
    def test_action_mask_picked(self):
        env, _ = get_sample_env()
        card1_idx = card_to_idx(c('2♣'))
        env.step(card1_idx)
        action_mask = env.action_masks()

        self.assertFalse(action_mask[card1_idx], 'Already picked cards should be illegal')
        self.assertEqual(1, np.sum(action_mask), 'There should only be 1 valid play')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣']), [], [], []],
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
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣', '3♣', '4♣']), [], [], []],
    )
    def test_terminated_after_cards_passed(self):
        env, _ = get_sample_env()
        _, _, terminated1, _, _ = env.step(0)
        _, _, terminated2, _, _ = env.step(1)
        _, _, terminated3, _, _ = env.step(2)
        self.assertFalse(terminated1, 'Environment should not terminate before all 3 cards are picked')
        self.assertFalse(terminated2, 'Environment should not terminate before all 3 cards are picked')
        self.assertTrue(terminated3, 'Environment should terminate after all 3 cards are picked')

    @patch.multiple(
        HeartsCore,
        **get_mock_params_for_hearts_core(),
    )
    @patch.multiple(
        HeartsCardsPassEnvironment.State,
        hands=[cl(['2♣', '3♣', '4♣']), [], [], []],
    )
    def test_rewards(self):
        env, _ = get_sample_env()
        _, reward1, _, _, _ = env.step(0)
        _, reward2, _, _, _ = env.step(1)
        _, reward3, _, _, _ = env.step(2)
        self.assertEqual(0, reward1, 'Reward after the first step should be zero')
        self.assertEqual(0, reward2, 'Reward after the second step should be zero')
        # technically not but usually:
        self.assertNotEqual(0, reward3, 'Reward after the third step should be non-zero')
