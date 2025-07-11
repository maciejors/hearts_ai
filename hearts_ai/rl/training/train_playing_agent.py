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
    create_agent,
    create_eval_callback,
    EPISODE_LENGTH_PLAY,
)
from .opponents.callbacks import (
    get_callback_from_agent,
    get_random_action_taking_callback,
    rule_based_play_callback, rule_based_card_pass_callback,
)


def train_playing_agent(
        agent_cls: Type[SupportedAlgorithm],
        env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_card_passing_agent: SupportedAlgorithm | None = None,
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
        eval_card_passing_agent: the trained card passing agent used for evaluations.
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
    agent = create_agent(agent_cls, env, seed=get_seed())

    if eval_card_passing_agent is not None:
        card_passing_callbacks = {
            'random': [
                get_callback_from_agent(eval_card_passing_agent),
                *[get_random_action_taking_callback(random_state=get_seed()) for _ in range(3)],
            ],
            'rule_based': [
                get_callback_from_agent(eval_card_passing_agent),
                *[rule_based_card_pass_callback for _ in range(3)],
            ],
        }
    else:
        card_passing_callbacks = {
            'random': None,
            'rule_based': None,
        }

    # sparse setting, because we only care about the end-of-round score
    eval_random_callback = create_eval_callback(
        HeartsPlayEnvironment(
            opponents_callbacks=[
                get_random_action_taking_callback(random_state=get_seed())
                for _ in range(3)
            ],
            card_passing_callbacks=card_passing_callbacks['random'],
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
            card_passing_callbacks=card_passing_callbacks['rule_based'],
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
        print(f'\n## Beginning stage {stage_no} out of {len(steps_per_stage)}')

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

    open(os.path.join(log_path, 'finished'), 'a').close()
    return agent
