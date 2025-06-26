import copy
import io
import pathlib
from typing import Union, Optional, Iterable, Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch as th
import torch.optim as optim
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm, BaseCallback
from stable_baselines3.common.save_util import (
    load_from_zip_file, save_to_zip_file
)
from stable_baselines3.common.type_aliases import MaybeCallback, GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv
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
        self._setup_model()

        self.n_episodes = n_episodes
        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_size = batch_size
        self._np_random = np.random.default_rng(seed)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def env_single(self) -> gym.Env:
        assert isinstance(self.env, DummyVecEnv)
        return self.env.envs[0]

    def __to_tensor(self, iterable: Iterable) -> torch.Tensor:
        return torch.tensor(np.array(iterable), dtype=torch.float32).to(self.device)

    def _setup_model(self) -> None:
        if not isinstance(self.env_single.action_space, gym.spaces.Discrete):
            raise ValueError('Only Discrete action space is supported')

        n_actions = self.env_single.action_space.n
        self.network = MCTSRLNetwork(  # type: ignore
            obs_size=self.env_single.observation_space.shape[0],
            n_actions=n_actions,
        ).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.mcts_rl_policy = MCTSRLPolicy(
            network=self.network,
            n_actions=n_actions,
            seed=self.seed,
            device=self.device,
        )

    def collect_samples(self, callback: BaseCallback) -> bool:
        """
        Returns:
            ``True`` if the training can be continued. ``False`` if callbacks require
            to stop training
        """
        for _ in range(self.n_episodes):
            obs = self.env_single.reset()[0]
            done = False
            episode_obs_and_policies = []
            episode_total_reward = 0

            while not done:
                action_masks = get_action_masks(self.env_single)
                action, policy_target = self.mcts_rl_policy.predict(
                    obs=obs,
                    env_deepcopy=copy.deepcopy(self.env_single),
                    deterministic=False,
                )
                episode_obs_and_policies.append((obs, policy_target))

                obs, reward, done, _, info = self.env_single.step(action)
                episode_total_reward += reward

                callback.update_locals(locals())
                if not callback.on_step():
                    return False

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
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        action, _ = self.mcts_rl_policy.predict(
            obs=observation,
            env_deepcopy=copy.deepcopy(self.env_single),
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
            learning_rate=data['learning_rate'],
            **kwargs,
        )
        model.buffer = data['buffer']
        model.network.load_state_dict(data['network_state_dict'])
        model.optimizer.load_state_dict(data['optimizer_state_dict'])
        return model
