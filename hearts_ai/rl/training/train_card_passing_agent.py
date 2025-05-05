import os
import warnings
from typing import Type

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.env import HeartsCardsPassEnvironment
from .common import (
    SupportedAlgorithm,
    update_self_play_clones,
    SaveAllRewards, get_random_action_taking_callback,
)


def train_card_passing_agent(
        agent_cls: Type[SupportedAlgorithm],
        playing_agent: SupportedAlgorithm,
        env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_freq_episodes: int = 600,
        n_eval_episodes: int = 500,
        progress_bar: bool = False,
        random_state: int | None = None,
) -> SupportedAlgorithm:
    """
    Args:
        agent_cls: Agent class
        playing_agent: A trained playing agent
        env_kwargs: Extra keyword arguments for :class:`HeartsCardPassEnvironment`
            (except for ``opponents_callbacks`` and ``playing_callbacks``)
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
        >>> agent = train_card_passing_agent(
        >>>     agent_cls=MaskablePPO,
        >>>     playing_agent=MaskablePPO.load('path/to/trained/player.zip'),
        >>>     env_kwargs={'eval_count': 10},
        >>>     log_path='training_logs',
        >>>     stages_lengths_episodes=[512*10, 512*10, 512*20, 512*25],
        >>>     eval_freq_episodes=600,
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

    ep_length = 3

    def playing_callback(play_obs, play_action_masks):
        return playing_agent.predict(play_obs, action_masks=play_action_masks)[0]

    env = HeartsCardsPassEnvironment(
        opponents_callbacks=[],
        playing_callbacks=playing_callback,
        **env_kwargs,
    )

    if agent_cls == MaskablePPO:
        n_steps = 1536  # multiple of 192 = 64 * 3 (batch_size * episode_length)
        print(f'PPO agent will update every {n_steps // ep_length} episodes')
        agent = MaskablePPO('MlpPolicy', env, n_steps=n_steps, seed=get_seed())
    else:
        raise ValueError('Unsupported agent_cls value. Use MaskablePPO')

    os.makedirs(log_path, exist_ok=True)

    # in evaluation we are only concerned about the end result of the round, hence the sparse setting.
    env_eval_random = Monitor(HeartsCardsPassEnvironment(
        opponents_callbacks=[get_random_action_taking_callback(random_state=get_seed())
                             for _ in range(3)],
        playing_callbacks=playing_callback,
    ))
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

    for stage_no, total_timesteps in enumerate(steps_per_stage.tolist(), 1):
        update_self_play_clones(env, agent)
        env.reset(seed=get_seed())

        log_subpath = os.path.join(log_path, f'stage_{stage_no}')
        logger = configure_sb3logger(log_subpath, ['csv'])
        agent.set_logger(logger)

        log_callback = LogEveryNTimesteps(1)
        save_all_rewards_callback = SaveAllRewards(log_subpath)

        callbacks = CallbackList([
            eval_random_callback,
            log_callback,
            save_all_rewards_callback,
        ])

        agent.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=progress_bar,
        )
    return agent
