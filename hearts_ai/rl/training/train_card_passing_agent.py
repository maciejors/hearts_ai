import os
from typing import Type

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger

from hearts_ai.rl.env import HeartsCardsPassEnvironment
from .common import (
    print_start_training_info,
    SupportedAlgorithm,
    update_self_play_clones,
    pre_train_setup,
    create_agent,
    create_eval_callback,
    EPISODE_LENGTH_CARD_PASS,
)
from .opponents.callbacks import (
    get_callback_from_agent,
    get_random_action_taking_callback,
    rule_based_card_pass_callback,
)


def train_card_passing_agent(
        agent_cls: Type[SupportedAlgorithm],
        playing_agent: SupportedAlgorithm,
        env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_freq_episodes: int = 2500,
        n_eval_episodes: int = 100,
        progress_bar: bool = False,
        random_state: int | None = None,
) -> SupportedAlgorithm:
    """
    A single run for the card passing agent. Run in a loop to record multiple
    runs.

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
            Whether to display progress bars during training. This is
            passed down to the models.
        random_state: Randomness control.

    Returns:
        A trained agent

    Examples:
        >>> card_passing_model = train_card_passing_agent(
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
          frequency. It is set to 512 episodes.
          Otherwise, PPO will extend the training to fill its
          buffer anyway.
    """
    get_seed, log_path = pre_train_setup(log_path, random_state)

    playing_callback = get_callback_from_agent(playing_agent)
    env = HeartsCardsPassEnvironment(
        opponents_callbacks=[],
        playing_callbacks=playing_callback,
        **env_kwargs,
    )
    agent = create_agent(agent_cls, env, seed=get_seed())

    eval_random_callback = create_eval_callback(
        HeartsCardsPassEnvironment(
            opponents_callbacks=[
                get_random_action_taking_callback(random_state=get_seed())
                for _ in range(3)
            ],
            playing_callbacks=playing_callback,
        ),
        eval_log_path=os.path.join(log_path, 'eval_random'),
        eval_freq=eval_freq_episodes * EPISODE_LENGTH_CARD_PASS,
        n_eval_episodes=n_eval_episodes,
        env_reset_seed=get_seed(),
    )
    eval_rule_based_callback = create_eval_callback(
        HeartsCardsPassEnvironment(
            opponents_callbacks=[rule_based_card_pass_callback] * 3,
            playing_callbacks=playing_callback,
        ),
        eval_log_path=os.path.join(log_path, 'eval_rule_based'),
        eval_freq=eval_freq_episodes * EPISODE_LENGTH_CARD_PASS,
        n_eval_episodes=n_eval_episodes,
        env_reset_seed=get_seed(),
    )

    steps_per_stage = np.array(stages_lengths_episodes) * EPISODE_LENGTH_CARD_PASS
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
