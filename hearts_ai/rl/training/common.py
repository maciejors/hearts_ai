import copy
import os
import re
import time
import warnings
from datetime import datetime
from typing import TypeVar, Callable, Type

import numpy as np
import torch
from gymnasium.core import ObsType, ActType
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.env import HeartsPlayEnvironment, HeartsCardsPassEnvironment
from hearts_ai.rl.env.utils import ActionTakingCallback

EPISODE_LENGTH_PLAY = 13
EPISODE_LENGTH_CARD_PASS = 3
STATS_WINDOW_SIZE_PLAY = 2000
STATS_WINDOW_SIZE_CARD_PASS = 1000
PPO_N_STEPS_PLAY = 2496  # 3x multiple of 832 = 64 * 13 (batch_size * episode_length)
PPO_N_STEPS_CARD_PASS = 1536  # 8x multiple of 192 = 64 * 3 (batch_size * episode_length)
MCTS_RL_N_EPISODES_PLAY = 192  # 3x multiple of batch_size = 64
MCTS_RL_N_EPISODES_CARD_PASS = 512  # 8x multiple of batch_size = 64
MCTS_RL_BUFFER_SIZE_PLAY = 512  # between 2-3x updates
MCTS_RL_BUFFER_SIZE_CARD_PASS = 1024  # 2x updates

SupportedAlgorithm = TypeVar(
    'SupportedAlgorithm',
    MaskablePPO,
    MaskableMCTSRL,
)
SupportedEnvironment = TypeVar(
    'SupportedEnvironment',
    HeartsPlayEnvironment,
    HeartsCardsPassEnvironment,
)


def print_start_training_info(steps_per_stage: np.ndarray):
    print(f'The training starts at {datetime.now().strftime("%H:%M")}')
    print(f'It will take {int(np.sum(steps_per_stage))} steps in total')


def _get_clone_callback(agent: SupportedAlgorithm) -> ActionTakingCallback:
    if isinstance(agent, MaskablePPO):
        temp_filename = f'temp_{int(time.time() * 1000)}.zip'
        agent.save(temp_filename)
        agent_copy = agent.load(temp_filename, env=agent.env)
        os.remove(temp_filename)
        return lambda state, action_masks: agent_copy.predict(state, action_masks=action_masks)[0]

    if isinstance(agent, MaskableMCTSRL):
        # can't copy the whole MCTS mechanism, as that would lead to infinite loops
        # therefore opponents only use the network
        agent_network_copy = copy.deepcopy(agent.network).to(agent.device)

        def callback(obs: ObsType, action_masks: np.ndarray) -> ActType:
            obs_tensor = torch.tensor(obs, dtype=torch.float32) \
                .unsqueeze(0) \
                .to(agent.device)
            with torch.no_grad():
                policy_logits, _ = agent_network_copy(obs_tensor)

            policy_logits_np = policy_logits.cpu().numpy()[0]
            policy_logits_np[~action_masks] = -np.inf
            action = np.argmax(policy_logits_np)
            return action

        return callback

    raise TypeError('Unsupported agent type')


def update_self_play_clones(env: SupportedEnvironment, agent: SupportedAlgorithm) -> None:
    env.opponents_callbacks = [_get_clone_callback(agent) for _ in range(3)]


def _get_next_run_dir(log_folder: str) -> str:
    """
    This function checks how many runs are already saved in the specified
    folder, and returns a full path for the logs for the next run.

    Returns:
        Absolute path to a directory for the next run.
    """
    if not os.path.isdir(log_folder):
        curr_max_run_no = 0
    else:
        curr_max_run_no = 0
        for subfolder_name in os.listdir(log_folder):
            if regex := re.fullmatch(r'run_(\d+)', subfolder_name):
                found_run_no = int(regex.group(1))
                curr_max_run_no = max(curr_max_run_no, found_run_no)

    next_run_subfolder = f'run_{curr_max_run_no + 1}'
    return os.path.abspath(os.path.join(log_folder, next_run_subfolder))


def pre_train_setup(log_path: str, random_state: int | None) -> tuple[Callable[[], int], str]:
    """
    Shared pre-training operations

    Returns:
        (get_seed callable, absolute log_path with appended run folder)
    """
    if random_state is None:
        warnings.warn(
            '`random_state` not set - consider using it for reproducibility',
            UserWarning,
        )
    rng = np.random.default_rng(random_state)

    def get_seed() -> int:
        return int(rng.integers(999999))

    run_log_path = _get_next_run_dir(log_path)
    os.makedirs(run_log_path, exist_ok=True)
    print(f'Logging to {run_log_path}')

    return get_seed, run_log_path


def _create_ppo_agent(
        env: SupportedEnvironment,
        seed: int,
) -> MaskablePPO:
    if isinstance(env, HeartsPlayEnvironment):
        print(f'PPO playing agent will update every {PPO_N_STEPS_PLAY // EPISODE_LENGTH_PLAY} episodes')
        return MaskablePPO(
            'MlpPolicy', env,
            n_steps=PPO_N_STEPS_PLAY,
            stats_window_size=STATS_WINDOW_SIZE_PLAY,
            seed=seed,
        )
    if isinstance(env, HeartsCardsPassEnvironment):
        print(f'PPO card passing agent will update every {PPO_N_STEPS_CARD_PASS // EPISODE_LENGTH_CARD_PASS} episodes')
        return MaskablePPO(
            'MlpPolicy', env,
            n_steps=PPO_N_STEPS_CARD_PASS,
            stats_window_size=STATS_WINDOW_SIZE_CARD_PASS,
            seed=seed,
        )
    raise TypeError(f'Unsupported environment: {type(env)}')


def _create_mcts_rl_agent(
        env: SupportedEnvironment,
        seed: int,
) -> MaskableMCTSRL:
    if isinstance(env, HeartsPlayEnvironment):
        print(f'MCTS-RL playing agent will update every {MCTS_RL_N_EPISODES_PLAY} episodes')
        return MaskableMCTSRL(
            env,
            n_episodes=MCTS_RL_N_EPISODES_PLAY,
            stats_window_size=STATS_WINDOW_SIZE_PLAY,
            buffer_size=MCTS_RL_BUFFER_SIZE_PLAY,
            seed=seed,
        )
    if isinstance(env, HeartsCardsPassEnvironment):
        print(f'MCTS-RL card passing agent will update every {MCTS_RL_N_EPISODES_CARD_PASS} episodes')
        return MaskableMCTSRL(
            env,
            n_episodes=MCTS_RL_N_EPISODES_CARD_PASS,
            stats_window_size=STATS_WINDOW_SIZE_CARD_PASS,
            buffer_size=MCTS_RL_BUFFER_SIZE_CARD_PASS,
            seed=seed,
        )
    raise TypeError(f'Unsupported environment: {type(env)}')


def create_agent(
        agent_cls: Type[SupportedAlgorithm],
        env: SupportedEnvironment,
        seed: int,
) -> SupportedAlgorithm:
    if agent_cls == MaskablePPO:
        return _create_ppo_agent(env, seed)
    if agent_cls == MaskableMCTSRL:
        return _create_mcts_rl_agent(env, seed)
    raise ValueError('Unsupported agent_cls value. Use MaskablePPO or MaskableMCTSRL')


def create_eval_callback(
        env: SupportedEnvironment,
        eval_log_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        env_reset_seed: int,
) -> MaskableEvalCallback:
    env_monitor = Monitor(env, info_keywords=("is_success",))
    env_monitor.reset(seed=env_reset_seed)
    return MaskableEvalCallback(
        env_monitor,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )
