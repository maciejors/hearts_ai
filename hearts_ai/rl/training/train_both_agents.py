import os
import warnings
from typing import Type

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import LogEveryNTimesteps, CallbackList
from stable_baselines3.common.logger import configure as configure_sb3logger
from stable_baselines3.common.monitor import Monitor

from hearts_ai.rl.env import HeartsPlayEnvironment, HeartsCardsPassEnvironment
from .common import (
    print_start_training_info,
    SupportedAlgorithm,
    update_self_play_clones,
)
from .opponents.callbacks import (
    get_callback_from_agent,
    get_random_action_taking_callback,
)


def train_both_agents(
        agent_cls: Type[SupportedAlgorithm],
        play_env_kwargs: dict,
        card_pass_env_kwargs: dict,
        log_path: str,
        stages_lengths_episodes: list[int],
        eval_freq_episodes: int = 10000,
        n_eval_episodes: int = 10000,
        progress_bar: bool = False,
        random_state: int | None = None,
) -> tuple[SupportedAlgorithm, SupportedAlgorithm]:
    """
    Trains the playing agent together with the card passing agent. They are evaluated
    together on a playing environment with card passing on. Only logs from the
    playing environment training are saved, although note that both agents
    participate in that training.

    Args:
        agent_cls: Agent class
        play_env_kwargs: Extra keyword arguments for :class:`HeartsPlayEnvironment`
            (except for ``opponents_callbacks``)
        card_pass_env_kwargs: Extra keyword arguments for :class:`HeartsCardPassEnvironment`
            (except for ``opponents_callbacks`` and ``playing_callbacks``)
        log_path: Path to a directory where logs and agents will be saved
        stages_lengths_episodes: Specifies how long the learning process
            should take, and when the opponents are updated with new clones of
            the learning agents. A value of ``[k1, k2]`` means the opponents
            will update after ``k1`` episodes of training, and then the
            training will proceed for another ``k2`` episodes before terminating.
            If this is a one-element list, the opponents will not be updated
            at all during training.
        eval_freq_episodes: how often to evaluate the agents (in episodes).
        n_eval_episodes: how many episodes to run in a single evaluation.
        progress_bar:
            Whether to display progress bars during training. This is
            passed down to the models.
        random_state: Randomness control.

    Returns:
        A trained agent

    Examples:
        >>> playing_model, card_passing_model = train_both_agents(
        >>>     agent_cls=MaskablePPO,
        >>>     play_env_kwargs={'reward_setting': 'dense'},
        >>>     card_pass_env_kwargs={'eval_count': 10},
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

    play_ep_length = 13
    card_pass_ep_length = 3

    play_env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        card_passing_callbacks=[],
        **play_env_kwargs,
    )
    card_pass_env = HeartsCardsPassEnvironment(
        opponents_callbacks=[],
        playing_callbacks=[],
        **card_pass_env_kwargs,
    )

    if agent_cls == MaskablePPO:
        play_n_steps = 2496  # 3x multiple of 832 = 64 * 13 (batch_size * episode_length)
        print(f'PPO playing agent will update every {play_n_steps // play_ep_length} episodes')
        playing_agent = MaskablePPO(
            'MlpPolicy', play_env,
            n_steps=play_n_steps,
            stats_window_size=2000,
            seed=get_seed(),
        )
        card_pass_n_steps = 1536  # 8x multiple of 192 = 64 * 3 (batch_size * episode_length)
        print(f'PPO card passing agent will update every {card_pass_n_steps // card_pass_ep_length} episodes')
        card_passing_agent = MaskablePPO(
            'MlpPolicy', card_pass_env,
            n_steps=card_pass_n_steps,
            stats_window_size=200,
            seed=get_seed(),
        )
    else:
        raise ValueError('Unsupported agent_cls value. Use MaskablePPO')

    os.makedirs(log_path, exist_ok=True)
    play_env.card_passing_callbacks = [get_callback_from_agent(card_passing_agent)]
    card_pass_env.playing_callbacks = [get_callback_from_agent(playing_agent)]

    # sparse setting, because we only care about the end-of-round score
    play_env_eval_random = Monitor(
        HeartsPlayEnvironment(
            opponents_callbacks=[
                get_random_action_taking_callback(random_state=get_seed())
                for _ in range(3)
            ],
            card_passing_callbacks=[
                get_callback_from_agent(card_passing_agent),
                *[get_random_action_taking_callback(random_state=get_seed())
                  for _ in range(3)],
            ],
            reward_setting='sparse',
        ),
        info_keywords=('is_success',),
    )
    play_env_eval_random.reset(seed=get_seed())

    eval_log_path = os.path.join(log_path, 'eval')
    eval_random_callback = MaskableEvalCallback(
        play_env_eval_random,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=eval_freq_episodes * play_ep_length,
        n_eval_episodes=n_eval_episodes,
    )

    steps_per_stage = np.array(stages_lengths_episodes) * play_ep_length
    print_start_training_info(steps_per_stage)

    for stage_no, stage_play_timesteps in enumerate(steps_per_stage.tolist(), 1):
        print(f'## Beginning stage {stage_no} out of {len(steps_per_stage)}')

        update_self_play_clones(play_env, playing_agent)
        update_self_play_clones(card_pass_env, card_passing_agent)
        play_env.card_passing_callbacks[1:] = card_pass_env.opponents_callbacks
        card_pass_env.playing_callbacks[1:] = play_env.opponents_callbacks

        play_env.reset(seed=get_seed())
        card_pass_env.reset(seed=get_seed())

        log_subpath = os.path.join(log_path, f'stage_{stage_no}')
        logger = configure_sb3logger(log_subpath, ['csv'])
        playing_agent.set_logger(logger)

        log_callback = LogEveryNTimesteps(1)

        callbacks = CallbackList([
            eval_random_callback,
            log_callback,
        ])

        # number of training timesteps for playing agent before card passing agent is tuned
        card_pass_tune_interval = play_n_steps * 50
        n_card_pass_tunes = stage_play_timesteps // card_pass_tune_interval
        # in case card_pass_tune_interval is not a multiple of stage_play_timesteps
        extra_playing_timesteps_after = stage_play_timesteps - card_pass_tune_interval * n_card_pass_tunes
        n_play_trainings = n_card_pass_tunes + (extra_playing_timesteps_after > 0)

        for swap_no in range(1, n_card_pass_tunes + 1):
            print(f'-> Training playing agent ({swap_no}/{n_play_trainings})')
            play_env.card_passing_callbacks[0] = get_callback_from_agent(card_passing_agent)
            playing_agent.learn(
                total_timesteps=card_pass_tune_interval,
                reset_num_timesteps=False,
                callback=callbacks,
                progress_bar=progress_bar,
            )
            print(f'-> Switching to card passing agent ({swap_no}/{n_card_pass_tunes})')
            card_pass_env.playing_callbacks[0] = get_callback_from_agent(playing_agent)
            card_passing_agent.learn(
                total_timesteps=card_pass_n_steps,
                reset_num_timesteps=False,
                progress_bar=progress_bar,
            )

        if extra_playing_timesteps_after > 0:
            print(f'-> Training playing agent ({n_play_trainings}/{n_play_trainings})')
            play_env.card_passing_callbacks[0] = get_callback_from_agent(card_passing_agent)
            playing_agent.learn(
                total_timesteps=extra_playing_timesteps_after,
                reset_num_timesteps=False,
                callback=callbacks,
                progress_bar=progress_bar,
            )

    return playing_agent, card_passing_agent
