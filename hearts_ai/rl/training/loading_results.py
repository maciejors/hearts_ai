import os

import numpy as np
import pandas as pd


def load_eval_results(log_path: str) -> pd.DataFrame:
    eval_results_path = os.path.join(log_path, 'eval', 'evaluations.npz')
    eval_results = np.load(eval_results_path)

    eval_no_col = []
    train_timestep_col = []
    stage_no_col = []
    episode_no_col = []
    reward_col = []

    stage_no = 1
    train_timestep_curr_stage_start = 0  # timestep at the start of the current stage

    timesteps_arr_orig = eval_results['timesteps']

    for eval_id, train_timestep_stage_relative in enumerate(timesteps_arr_orig):
        # check for the next stage
        if len(train_timestep_col) > 0:
            if train_timestep_stage_relative <= timesteps_arr_orig[eval_id - 1]:
                stage_no += 1
                train_timestep_curr_stage_start = train_timestep_col[-1]
        
        train_timestep_absolute = train_timestep_curr_stage_start + train_timestep_stage_relative

        episodes_rewards = eval_results['results'][eval_id]
        n_episodes = len(episodes_rewards)

        eval_no_col.extend([eval_id+1] * n_episodes)
        train_timestep_col.extend([train_timestep_absolute] * n_episodes)
        stage_no_col.extend([stage_no] * n_episodes)
        episode_no_col.extend([i+1 for i in range(n_episodes)])
        reward_col.extend(episodes_rewards)

    eval_results_df = pd.DataFrame({
        'eval_no': eval_no_col,
        'train_timestep': train_timestep_col,
        'stage': stage_no_col,
        'episode': episode_no_col,
        'reward': reward_col,
    })
    return eval_results_df


def load_all_training_rewards(log_path: str) -> pd.DataFrame:
    all_dfs = [] 
    stage_step_add = 0  # how much to add to stage step to obtain absolute step

    for stage_subdir in os.listdir(log_path):
        if not stage_subdir.startswith('stage_'):
            continue

        stage_no = int(stage_subdir.split('_')[-1])
        stage_rewards_df = pd.read_csv(
            os.path.join(log_path, stage_subdir, 'rewards_all.csv')
        )
        stage_rewards_df['stage'] = stage_no
        stage_rewards_df['step_abs'] = stage_rewards_df['step'].copy()
        stage_rewards_df['step_abs'] += stage_step_add

        stage_step_add = np.max(stage_rewards_df['step_abs'])
        all_dfs.append(stage_rewards_df)

    all_rewards_df = pd.concat(all_dfs, ignore_index=True)
    return all_rewards_df


def load_training_logs(log_path: str) -> pd.DataFrame:
    all_dfs = [] 
    stage_step_add = 0  # how much to add to time/total_timesteps to obtain absolute timestep

    for stage_subdir in os.listdir(log_path):
        if not stage_subdir.startswith('stage_'):
            continue

        stage_no = int(stage_subdir.split('_')[-1])
        raw_logs_df = pd.read_csv(
            os.path.join(log_path, stage_subdir, 'progress.csv')
        )

        # filter out entries with algo update info, and entries with eval info
        raw_logs_df = raw_logs_df[
            (raw_logs_df['time/iterations'].isna())
            & (~raw_logs_df['rollout/ep_rew_mean'].isna())
        ]

        stage_training_logs_df = pd.DataFrame({
            'fps': raw_logs_df['time/fps'],
            'timestep': raw_logs_df['time/total_timesteps'],
            'timestep_abs': raw_logs_df['time/total_timesteps'] + stage_step_add,
            'ep_rew_mean': raw_logs_df['rollout/ep_rew_mean'],
        })
        stage_training_logs_df['stage'] = stage_no

        ep_length = int(raw_logs_df['rollout/ep_len_mean'].iloc[1])
        stage_training_logs_df['episode'] = stage_training_logs_df['timestep'] // ep_length
        stage_training_logs_df['episode_abs'] = stage_training_logs_df['timestep_abs'] // ep_length

        stage_step_add = np.max(stage_training_logs_df['timestep_abs'])
        all_dfs.append(stage_training_logs_df)

    training_logs_df = pd.concat(all_dfs, ignore_index=True)
    return training_logs_df


def load_all_results(log_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all results in order: Evaluation results, All training rewards, Training logs
    """
    return (
        load_eval_results(log_path),
        load_all_training_rewards(log_path),
        load_training_logs(log_path),
    )
