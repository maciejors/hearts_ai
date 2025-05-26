import os
from typing import Type

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger

from hearts_ai.rl.env import HeartsPlayEnvironment
from .common import (
    print_start_training_info,
    SupportedAlgorithm,
    update_self_play_clones,
    pre_train_setup,
    create_eval_callback,
    EPISODE_LENGTH_PLAY,
    PPO_N_STEPS_PLAY,
    STATS_WINDOW_SIZE_PLAY,
)
from .opponents.callbacks import (
    get_random_action_taking_callback,
    rule_based_play_callback,
)


def train_playing_agent(
        agent_cls: Type[SupportedAlgorithm],
        env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_freq_episodes: int = 10000,
        n_eval_episodes: int = 10000,
        progress_bar: bool = False,
        random_state: int | None = None,
) -> SupportedAlgorithm:
    """
    A single run for the playing agent. Run in a loop to record multiple
    runs.

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
            Whether to display progress bars during training. This is
            passed down to the models.
        random_state: Randomness control.

    Returns:
        A trained agent

    Examples:
        >>> playing_model = train_playing_agent(
        >>>     agent_cls=MaskablePPO,
        >>>     env_kwargs={'reward_setting': 'dense'},
        >>>     log_path='training_logs',
        >>>     stages_lengths_episodes=[192*10, 192*10, 192*20, 192*25],
        >>>     eval_freq_episodes=200,
        >>>     n_eval_episodes=400,
        >>>     progress_bar=True,
        >>>     random_state=42,
        >>> )

    Notes:
        - For PPO it is recommended to set :param:`opponents_update_freq`
          values and :param:`final_stage_len` to a multiple of PPO's update
          frequency. It is set to 192 episodes.
          Otherwise, PPO will extend the training to fill its buffer anyway.
    """
    get_seed, log_path = pre_train_setup(log_path, random_state)

    env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        **env_kwargs,
    )

    if agent_cls == MaskablePPO:
        print(f'PPO agent will update every {PPO_N_STEPS_PLAY // EPISODE_LENGTH_PLAY} episodes')
        agent = MaskablePPO(
            'MlpPolicy', env,
            n_steps=PPO_N_STEPS_PLAY,
            stats_window_size=STATS_WINDOW_SIZE_PLAY,
            seed=get_seed(),
        )
    else:
        raise ValueError('Unsupported agent_cls value. Use MaskablePPO')

    # sparse setting, because we only care about the end-of-round score
    eval_random_callback = create_eval_callback(
        HeartsPlayEnvironment(
            opponents_callbacks=[
                get_random_action_taking_callback(random_state=get_seed())
                for _ in range(3)
            ],
            reward_setting='sparse',
        ),
        eval_log_path=os.path.join(log_path, 'eval_random'),
        eval_freq=eval_freq_episodes * EPISODE_LENGTH_PLAY,
        n_eval_episodes=n_eval_episodes,
        env_reset_seed=get_seed(),
    )
    eval_rule_based_callback = create_eval_callback(
        HeartsPlayEnvironment(
            opponents_callbacks=[rule_based_play_callback] * 3,
            reward_setting='sparse',
        ),
        eval_log_path=os.path.join(log_path, 'eval_rule_based'),
        eval_freq=eval_freq_episodes * EPISODE_LENGTH_PLAY,
        n_eval_episodes=n_eval_episodes,
        env_reset_seed=get_seed(),
    )

    steps_per_stage = np.array(stages_lengths_episodes) * EPISODE_LENGTH_PLAY
    print_start_training_info(steps_per_stage)

    for stage_no, stage_timesteps in enumerate(steps_per_stage.tolist(), 1):
        print(f'## Beginning stage {stage_no} out of {len(steps_per_stage)}')

        update_self_play_clones(env, agent)
        env.reset(seed=get_seed())

        log_subpath = os.path.join(log_path, f'stage_{stage_no}')
        logger = configure_sb3logger(log_subpath, ['csv'])
        agent.set_logger(logger)

        callbacks = CallbackList([
            LogEveryNTimesteps(1),
            eval_random_callback,
            eval_rule_based_callback,
        ])
        agent.learn(
            total_timesteps=stage_timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=progress_bar,
        )
    return agent
