import os
import re
import time
import warnings
from datetime import datetime
from typing import TypeVar, Callable

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.env import HeartsPlayEnvironment, HeartsCardsPassEnvironment

EPISODE_LENGTH_PLAY = 13
EPISODE_LENGTH_CARD_PASS = 3
STATS_WINDOW_SIZE_PLAY = 2000
STATS_WINDOW_SIZE_CARD_PASS = 500
PPO_N_STEPS_PLAY = 2496  # 3x multiple of 832 = 64 * 13 (batch_size * episode_length)
PPO_N_STEPS_CARD_PASS = 1536  # 8x multiple of 192 = 64 * 3 (batch_size * episode_length)


SupportedAlgorithm = TypeVar(
    'SupportedAlgorithm',
    MaskablePPO,
    MaskablePPO,  # this is placeholder, and will be replaced with a new algorithm later
)
SupportedEnvironment = TypeVar(
    'SupportedEnvironment',
    HeartsPlayEnvironment,
    HeartsCardsPassEnvironment,
)


def print_start_training_info(steps_per_stage: np.ndarray):
    print(f'The training starts at {datetime.now().strftime("%H:%M")}')
    print(f'It will take {int(np.sum(steps_per_stage))} steps in total')


def clone_agent(agent: SupportedAlgorithm) -> SupportedAlgorithm:
    temp_filename = f'temp_{int(time.time() * 1000)}.zip'
    agent.save(temp_filename)
    agent_copy = agent.load(temp_filename)
    os.remove(temp_filename)
    return agent_copy


def update_self_play_clones(env: SupportedEnvironment, agent: SupportedAlgorithm) -> None:
    agent_copy = clone_agent(agent)
    opponents_callbacks = [
        lambda state, action_masks: agent_copy.predict(state, action_masks=action_masks)[0]
        for _ in range(3)
    ]
    env.opponents_callbacks = opponents_callbacks


def _get_next_run_dir(log_folder: str) -> str:
    """
    This function checks how many runs are already saved in the specified
    folder, and returns a full path for the logs for the next run.

    Returns:
        Absolute path to a directory for the next run.
    """
    if not os.path.isdir(log_folder):
        next_run_no = 1
    else:
        next_run_no = 1
        for subfolder_name in os.listdir(log_folder):
            if regex := re.fullmatch(r'run_(\d+)', subfolder_name):
                found_run_no = int(regex.group(1))
                next_run_no = max(next_run_no, found_run_no)

    next_run_subfolder = f'run_{next_run_no + 1}'
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

    log_path = _get_next_run_dir(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f'Logging to {log_path}')

    return get_seed, log_path


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
