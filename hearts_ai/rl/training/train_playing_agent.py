import os.path
from typing import Literal, Type

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.env import HeartsPlayEnvironment
from .common import (
    SupportedAlgorithm,
    update_self_play_clones,
    SaveAllRewards, get_random_action_taking_callback,
)


def train_playing_agent(
        agent_cls: Type[SupportedAlgorithm],
        reward_setting: Literal['dense', 'sparse'],
        opponents_update_freq: list[int],
        final_stage_len: int,
        log_path: str,
        progress_bar: bool = False,
) -> SupportedAlgorithm:
    """
    Args:
        agent_cls: Agent class
        reward_setting: Reward setting for the playing environment
        log_path: Path to a directory where logs and agents will be saved
        opponents_update_freq: Specifies how often opponents are updated with
            new clones of the learning agent. A value of ``k`` means the
            opponents will update every ``k`` updates of the agent. If empty,
            the opponents will not be updated at all during training.
        final_stage_len: For how many updates of the agent to run the final
            stage of training (i.e. after the final opponents' update)
        progress_bar:
            Whether or not to display progress bars during training. This is
            passed down to the models.

    Returns:
        A trained agent

    Examples:
        >>> agent = train_playing_agent(
        >>>     agent_cls=MaskablePPO,
        >>>     reward_setting='dense',
        >>>     log_path='training_logs',
        >>>     opponents_update_freq=[50, 100, 100],
        >>>     final_stage_len=250,
        >>>     progress_bar=True,
        >>> )
    """
    env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        reward_setting=reward_setting,
    )
    if agent_cls == MaskablePPO:
        n_steps = 2496  # multiple of 832 = 64 * 13 (batch_size * episode_length)
        agent = MaskablePPO('MlpPolicy', env, n_steps=n_steps, seed=28)
    else:
        raise ValueError('Unsupported agent_cls value. Use MaskablePPO')

    os.makedirs(log_path, exist_ok=True)

    # in evaluation we are only concerned about the end result of the round, hence the sparse setting.
    env_eval_random = Monitor(HeartsPlayEnvironment(
        opponents_callbacks=[get_random_action_taking_callback(random_state=i)
                             for i in range(3)],
        reward_setting='sparse',
    ))
    eval_log_path = os.path.join(log_path, 'eval')
    eval_random_callback = MaskableEvalCallback(
        env_eval_random,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=n_steps,
        n_eval_episodes=100,
    )

    steps_per_stage = np.array(opponents_update_freq) * n_steps
    steps_per_stage = np.append(steps_per_stage, final_stage_len * n_steps)

    for stage_no, total_timesteps in enumerate(steps_per_stage.tolist(), 1):
        update_self_play_clones(env, agent)

        log_subpath = os.path.join(log_path, f'stage_{stage_no}')
        logger = configure_sb3logger(log_subpath, ['csv'])
        agent.set_logger(logger)

        log_callback = LogEveryNTimesteps(13)  # ep length
        save_all_rewards_callback = SaveAllRewards(log_subpath)

        callbacks = CallbackList([
            eval_random_callback,
            log_callback,
            save_all_rewards_callback,
        ])
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
        )

    return agent
