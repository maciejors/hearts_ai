import os
import warnings
from typing import Type

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.env import HeartsPlayEnvironment
from hearts_ai.rl.env.opponents_callbacks import (
    get_random_action_taking_callback
)
from .common import (
    print_start_training_info,
    SupportedAlgorithm,
    update_self_play_clones,
)


def train_playing_agent(
        agent_cls: Type[SupportedAlgorithm],
        env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_freq_episodes: int = 200,
        n_eval_episodes: int = 250,
        progress_bar: bool = False,
        random_state: int | None = None,
) -> SupportedAlgorithm:
    """
    Args:
        agent_cls: Agent class
        env_kwargs: Extra keyword arguments for :class:`HeartsPlayEnvironment`
            (except for ``opponents_callbacks``)
        log_path: Path to a directory where logs and agents will be saved
        stages_lengths_episodes: Specifies how long the learning process
            should take, and when the opponents are updated with new clones of
            the learning agent. A value of ``[k1, k2]`` means the opponents
            will update after ``k1`` episodes of training, and then the
            training will proceed for another ``k2`` episodes before terminating.
            If this is a one-element list, the opponents will not be updated
            at all during training.
        eval_freq_episodes: how often to evaluate the agent (in episodes).
        n_eval_episodes: how many episodes to run in a single evaluation.
        progress_bar:
            Whether or not to display progress bars during training. This is
            passed down to the models.
        random_state: Randomness control.

    Returns:
        A trained agent

    Examples:
        >>> agent = train_playing_agent(
        >>>     agent_cls=MaskablePPO,
        >>>     env_kwargs={'reward_setting': 'dense'},
        >>>     log_path='training_logs',
        >>>     stages_lengths_episodes=[192*10, 192*10, 192*20, 192*25],
        >>>     eval_freq_episodes=200,
        >>>     progress_bar=True,
        >>>     random_state=42,
        >>> )

    Notes:
        - For PPO it is recommended to set :param:`opponents_update_freq`
          values and :param:`final_stage_len` to a multiple of PPO's update
          frequency. Otherwise PPO will extend the training to fill its
          buffer anyway.
    """
    if random_state is None:
        warnings.warn(
            '`random_state` not set - consider using it for reproducibility',
            UserWarning,
        )
    rng = np.random.default_rng(random_state)

    def get_seed() -> int:
        return int(rng.integers(999999))

    ep_length = 13

    env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        **env_kwargs,
    )

    if agent_cls == MaskablePPO:
        n_steps = 2496  # multiple of 832 = 64 * 13 (batch_size * episode_length)
        print(f'PPO agent will update every {n_steps // ep_length} episodes')
        agent = MaskablePPO(
            'MlpPolicy', env,
            n_steps=n_steps,
            stats_window_size=1000,
            seed=get_seed(),
        )
    else:
        raise ValueError('Unsupported agent_cls value. Use MaskablePPO')

    os.makedirs(log_path, exist_ok=True)

    # sparse setting, because we only care about the end-of-round score
    env_eval_random = Monitor(
        HeartsPlayEnvironment(
            opponents_callbacks=[get_random_action_taking_callback(random_state=get_seed())
                                 for _ in range(3)],
            reward_setting='sparse',
        ),
        info_keywords=("is_success",),
    )
    env_eval_random.reset(seed=get_seed())

    eval_log_path = os.path.join(log_path, 'eval')
    eval_random_callback = MaskableEvalCallback(
        env_eval_random,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=eval_freq_episodes * ep_length,
        n_eval_episodes=n_eval_episodes,
    )

    steps_per_stage = np.array(stages_lengths_episodes) * ep_length
    print_start_training_info(steps_per_stage)

    for stage_no, total_timesteps in enumerate(steps_per_stage.tolist(), 1):
        update_self_play_clones(env, agent)
        env.reset(seed=get_seed())

        log_subpath = os.path.join(log_path, f'stage_{stage_no}')
        logger = configure_sb3logger(log_subpath, ['csv'])
        agent.set_logger(logger)

        log_callback = LogEveryNTimesteps(1)

        callbacks = CallbackList([
            eval_random_callback,
            log_callback,
        ])

        agent.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=progress_bar,
        )
    return agent
