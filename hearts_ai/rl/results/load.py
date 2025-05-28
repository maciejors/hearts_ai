import os

import numpy as np
import pandas as pd

from .training_results import TrainingResults


def _load_eval_results_single_run(run_log_path: str, eval_id: str) -> pd.DataFrame:
    eval_results_path = os.path.join(run_log_path, f'eval_{eval_id}', 'evaluations.npz')
    eval_results = np.load(eval_results_path)

    eval_no_col = []
    train_timestep_col = []
    eval_episode_no_col = []
    reward_col = []
    is_success_col = []

    for eval_id, train_timestep in enumerate(eval_results['timesteps']):
        episodes_rewards = eval_results['results'][eval_id].astype(int)
        episodes_successes = eval_results['successes'][eval_id]
        n_episodes = len(episodes_rewards)

        eval_no_col.extend([eval_id + 1] * n_episodes)
        train_timestep_col.extend([train_timestep] * n_episodes)
        eval_episode_no_col.extend([i + 1 for i in range(n_episodes)])
        reward_col.extend(episodes_rewards)
        is_success_col.extend(episodes_successes)

    eval_results_df = pd.DataFrame({
        'eval_no': eval_no_col,
        'train_timestep': train_timestep_col,
        'eval_episode': eval_episode_no_col,
        'reward': reward_col,
        'is_success': is_success_col,
    })
    return eval_results_df


def _load_training_logs_single_run(run_log_path: str) -> tuple[pd.DataFrame, list[int]]:
    all_dfs = []

    for stage_subdir in os.listdir(run_log_path):
        if not stage_subdir.startswith('stage_'):
            continue

        stage_no = int(stage_subdir.split('_')[-1])
        raw_logs_df = pd.read_csv(
            os.path.join(run_log_path, stage_subdir, 'progress.csv')
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
    Loads all results from a single training process with multiple runs
    """
    training_logs_all_dfs: list[pd.DataFrame] = []
    eval_results_all_dfs: dict[str, list[pd.DataFrame]] = {
        'random': [],
        'rule_based': [],
    }
    stages_ends: list[int] = []

    for run_folder_name in os.listdir(log_path):
        run_log_path = os.path.join(log_path, run_folder_name)

        if not os.path.exists(os.path.join(run_log_path, 'finished')):
            continue  # means this run is in progress

        # stages_ends should be the same for every run
        training_logs_df, stages_ends = _load_training_logs_single_run(run_log_path)
        training_logs_df['run_id'] = run_folder_name
        training_logs_all_dfs.append(training_logs_df)

        for eval_id in eval_results_all_dfs:
            eval_results_df = _load_eval_results_single_run(run_log_path, eval_id)
            eval_results_df['run_id'] = run_folder_name
            eval_results_all_dfs[eval_id].append(eval_results_df)

    return TrainingResults(
        training_logs=pd.concat(training_logs_all_dfs, ignore_index=True),
        eval_results={
            eval_id: pd.concat(dataframe_list, ignore_index=True)
            for eval_id, dataframe_list in eval_results_all_dfs.items()
        },
        stages_ends=stages_ends,
    )
