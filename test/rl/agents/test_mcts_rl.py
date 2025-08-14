import os
import tempfile
import unittest

import numpy as np
import torch

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.env import HeartsPlayEnvironment


class TestMaskableMCTSRL(unittest.TestCase):
    def test_load_and_save(self):
        sample_env = HeartsPlayEnvironment(
            reward_setting='sparse',
            opponents_callbacks=[],
        )
        agent = MaskableMCTSRL(
            env=sample_env,
            n_episodes=123,
            buffer_size=456,
            n_simulations=11,
            max_tree_depth=101,
            batch_size=32,
            learning_rate=0.1,
            seed=0,
        )
        buffer_sample = (
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            13,
        )
        agent.buffer.append(buffer_sample)

        # act
        tmpfile = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        try:
            agent.save(tmpfile.name)
            sut = MaskableMCTSRL.load(tmpfile.name, env=sample_env)
        finally:
            tmpfile.close()
            os.remove(tmpfile.name)

        # assert
        self.assertEqual(123, sut.n_episodes)
        self.assertEqual(456, sut.buffer_size)
        self.assertEqual(32, sut.batch_size)
        self.assertEqual(0.1, sut.learning_rate)
        self.assertEqual(11, sut.n_simulations)
        self.assertEqual(101, sut.max_tree_depth)
        self.assertEqual(agent.device, sut.device)

        self.assertEqual(1, len(sut.buffer))
        sut_buffer_sample = sut.buffer[0]
        self.assertEqual(3, len(sut_buffer_sample))
        np.testing.assert_array_equal(buffer_sample[0], sut_buffer_sample[0])
        np.testing.assert_array_equal(buffer_sample[1], sut_buffer_sample[1])
        self.assertEqual(buffer_sample[2], sut_buffer_sample[2])

        sample_obs_tensor = torch.tensor(
            np.ones(shape=sample_env.observation_space.shape),
            dtype=torch.float32,
        ).to(agent.device)
        original_policy_pred, original_value_pred = agent.network(sample_obs_tensor)
        loaded_policy_pred, loaded_value_pred = sut.network(sample_obs_tensor)
        self.assertListEqual(original_policy_pred.tolist(), loaded_policy_pred.tolist())
        self.assertEqual(original_value_pred.item(), loaded_value_pred.item())
