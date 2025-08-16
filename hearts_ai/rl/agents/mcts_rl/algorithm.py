import io
import pathlib
import sys
import time
from typing import Union, Optional, Iterable, Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.optim as optim
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm, BaseCallback
from stable_baselines3.common.save_util import (
    load_from_zip_file, save_to_zip_file
)
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.utils import safe_mean
from torch.nn.functional import mse_loss, cross_entropy

from .network import MCTSRLNetwork
from .policy import MCTSRLPolicy


class MaskableMCTSRL(BaseAlgorithm):
    """
    AlphaZero-like algorithm implementation as a SB3 algorithm.
    This class extends BaseAlgorithm only to take advantage of callbacks and
    SB3's logging capabilities, as such parameters like learning_rate, and
    BaseAlgorithm's policy etc. are not used..

    Inner workings (like methods naming, handling callbacks) inspired by :class:`MaskablePPO`
    (https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_mask/ppo_mask.py)

    Note:
        This algorithm relies on making deep copies of the environment. It
        is recommended to implement __deepcopy__() method for the environment
        if that can improve performance.
    """

    def __init__(
            self,
            env: gym.Env,
            n_episodes: int = 192,
            buffer_size: int = 512,
            batch_size: int = 64,
            max_tree_depth: int | None = None,
            n_simulations: int = 50,
            learning_rate: float | Callable[[float], float] = 3e-4,
            seed: int | None = None,
            **kwargs,
    ):
        super().__init__(
            env=env,
            seed=seed,
            learning_rate=learning_rate,
            # The remaining parameters below are not used and are passed only
            # for compatibility with the superclass
            policy=MCTSRLNetwork,  # type: ignore
            **kwargs,
        )
        assert self.env.num_envs == 1, 'This algorithm does not support concurrent learning on multiple envs'

        self.n_episodes = n_episodes
        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_size = batch_size
        self.max_tree_depth = max_tree_depth
        self.n_simulations = n_simulations
        self._np_random = np.random.default_rng(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._setup_model()

    def __to_tensor(self, iterable: Iterable) -> torch.Tensor:
        return torch.tensor(np.array(iterable), dtype=torch.float32).to(self.device)

    def _setup_model(self) -> None:
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise ValueError('Only Discrete action space is supported')

        n_actions = self.env.action_space.n
        self.network = MCTSRLNetwork(  # type: ignore
            obs_size=self.env.observation_space.shape[0],
            n_actions=n_actions,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.mcts_rl_policy = MCTSRLPolicy(
            network=self.network,
            n_actions=n_actions,
            max_tree_depth=self.max_tree_depth,
            n_simulations=self.n_simulations,
            seed=self.seed,
            device=self.device,
        )

    def dump_logs(self) -> None:
        """
        Copied from:
        https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def collect_samples(self, callback: BaseCallback) -> bool:
        """
        Returns:
            ``True`` if the training can be continued. ``False`` if callbacks require
            to stop training
        """
        for _ in range(self.n_episodes):
            obs = self.env.reset()
            dones = [False]
            episode_obs_and_policies = []
            episode_total_reward = 0

            while not dones[0]:
                actions, policy_targets = self.mcts_rl_policy.predict(
                    obs=obs,
                    env=self.env,
                    deterministic=False,
                )
                episode_obs_and_policies.append((obs[0], policy_targets[0]))

                obs, rewards, dones, infos = self.env.step(actions)
                episode_total_reward += rewards[0]

                self.num_timesteps += 1

                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos, dones)

            for state, policy in episode_obs_and_policies:
                self.buffer.append((state, policy, episode_total_reward))

            if len(self.buffer) > self.buffer_size:
                self.buffer = self.buffer[-self.buffer_size:]
        return True

    def train(self):
        self.network.train()

        num_samples = len(self.buffer)
        samples_idx_shuffled = np.arange(num_samples)
        self._np_random.shuffle(samples_idx_shuffled)

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_idx = samples_idx_shuffled[start_idx:end_idx]
            batch = [self.buffer[i] for i in batch_idx]

            states, policy_targets, value_targets = zip(*batch)
            states = self.__to_tensor(states)
            policy_targets = self.__to_tensor(policy_targets)
            value_targets = self.__to_tensor(value_targets)

            policy_predictions, value_predictions = self.network(states)
            policy_loss = cross_entropy(policy_predictions, policy_targets)
            value_loss = mse_loss(value_predictions, value_targets)

            total_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def learn(
            self: SelfBaseAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ) -> SelfBaseAlgorithm:

        if not is_masking_supported(self.env):
            raise ValueError('The environment does not support action masking')

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            self.collect_samples(callback)
            self.train()

        callback.on_training_end()
        return self

    def predict(  # type: ignore[override]
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
            action_masks: Optional[np.ndarray] = None,
            sb3_eval_mode: bool = True,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation.

        Args:
            observation: the input observation
            state: unused parameter
            episode_start: unused parameter
            deterministic: Whether or not to return deterministic actions.
            action_masks: Mask indicating allowed actions
            sb3_eval_mode: If ``True``, only network prediction is used to decide
                an action. If ``False``, standard MCTS+RL prediction will be
                made. Default is ``True``, because in SB3 framework it is not
                possible to perform simulations inside EvalCallbacks

        Returns:
            the model's action and the next hidden state
            (the latter used in recurrent policies, here it is always None)
        """
        if sb3_eval_mode:
            policy_probs = self.mcts_rl_policy.get_network_policy(observation, action_masks)
            if deterministic:
                action = np.argmax(policy_probs)
            else:
                n_actions = len(policy_probs)
                action = self._np_random.choice(list(range(n_actions)), p=policy_probs)
        else:
            action, _ = self.mcts_rl_policy.predict(
                obs=observation,
                env=self.env,
                deterministic=deterministic,
            )
        return np.array([action]), None

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        data = {
            'n_episodes': self.n_episodes,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'max_tree_depth': self.max_tree_depth,
            'n_simulations': self.n_simulations,
            'learning_rate': self.learning_rate,
            'buffer': self.buffer,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        save_to_zip_file(path, data=data, params=None, pytorch_variables=None)

    @classmethod
    def load(  # noqa: C901
            cls: type[SelfBaseAlgorithm],
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            custom_objects: Optional[dict[str, Any]] = None,
            print_system_info: bool = False,
            force_reset: bool = True,
            **kwargs,
    ) -> SelfBaseAlgorithm:
        data, _, _ = load_from_zip_file(path)
        model = cls(
            env=env,
            n_episodes=data['n_episodes'],
            buffer_size=data['buffer_size'],
            batch_size=data['batch_size'],
            max_tree_depth=data['max_tree_depth'],
            n_simulations=data['n_simulations'],
            learning_rate=data['learning_rate'],
            **kwargs,
        )
        model.buffer = data['buffer']
        model.network.load_state_dict(data['network_state_dict'])
        model.optimizer.load_state_dict(data['optimizer_state_dict'])
        return model
