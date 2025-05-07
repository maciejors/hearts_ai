import os

import numpy as np
import pandas as pd

from .training_results import TrainingResults


def _load_eval_results(log_path: str) -> pd.DataFrame:
    eval_results_path = os.path.join(log_path, 'eval', 'evaluations.npz')
    eval_results = np.load(eval_results_path)

    eval_no_col = []
    train_timestep_col = []
    eval_episode_no_col = []
    points_col = []
    is_success_col = []

    for eval_id, train_timestep in enumerate(eval_results['timesteps']):
        episodes_points = -eval_results['results'][eval_id].astype(int)
        episodes_successes = eval_results['successes'][eval_id]
        n_episodes = len(episodes_points)

        eval_no_col.extend([eval_id + 1] * n_episodes)
        train_timestep_col.extend([train_timestep] * n_episodes)
        eval_episode_no_col.extend([i + 1 for i in range(n_episodes)])
        points_col.extend(episodes_points)
        is_success_col.extend(episodes_successes)

    eval_results_df = pd.DataFrame({
        'eval_no': eval_no_col,
        'train_timestep': train_timestep_col,
        'eval_episode': eval_episode_no_col,
        'points': points_col,
        'is_success': is_success_col,
    })
    return eval_results_df


def _load_training_logs(log_path: str) -> tuple[pd.DataFrame, list[int]]:
    all_dfs = []

    for stage_subdir in os.listdir(log_path):
        if not stage_subdir.startswith('stage_'):
            continue

        stage_no = int(stage_subdir.split('_')[-1])
        raw_logs_df = pd.read_csv(
            os.path.join(log_path, stage_subdir, 'progress.csv')
        )
        ep_length = int(raw_logs_df['rollout/ep_len_mean'].iloc[-1])

        # filter out entries with algo update info, and entries with eval info
        raw_logs_df = raw_logs_df[
            (raw_logs_df['time/iterations'].isna())
            & (~raw_logs_df['rollout/ep_rew_mean'].isna())
            & (raw_logs_df['time/total_timesteps'] % ep_length == 0)
            ]

        stage_training_logs_df = pd.DataFrame({
            'fps': raw_logs_df['time/fps'],
            'timestep': raw_logs_df['time/total_timesteps'],
            'ep_rew_mean': raw_logs_df['rollout/ep_rew_mean'],
            'success_rate': raw_logs_df['rollout/success_rate'],
        })
        stage_training_logs_df['stage'] = stage_no
        stage_training_logs_df['episode'] = stage_training_logs_df['timestep'] // ep_length

        all_dfs.append(stage_training_logs_df)

    training_logs_df = pd.concat(all_dfs, ignore_index=True)

    stages_ends = training_logs_df \
        .groupby('stage') \
        .max()['timestep'] \
        .tolist()

    return training_logs_df, stages_ends


def load_training_results(log_path: str) -> TrainingResults:
    """
    Loads all results from a single training process
    """
    training_logs_df, stages_ends = _load_training_logs(log_path)
    eval_results_df = _load_eval_results(log_path)

    return TrainingResults(
        eval_results_df=eval_results_df,
        training_logs_df=training_logs_df,
        stages_ends=stages_ends,
    )
