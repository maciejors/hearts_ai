from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class TrainingResults:
    """Results of a single training process"""
    eval_results: dict[str, pd.DataFrame]
    training_logs: pd.DataFrame
    stages_ends: list[int]


def print_best_eval_run(results: TrainingResults, eval_id: Literal['random', 'rule_based']) -> None:
    eval_df = results.eval_results[eval_id]
    eval_df = eval_df[['run_id', 'eval_no', 'train_timestep', 'reward']]

    eval_mean_df = eval_df \
        .groupby(['run_id', 'eval_no', 'train_timestep']) \
        .mean() \
        .reset_index()
    eval_std_df = eval_df \
        .groupby(['run_id', 'eval_no', 'train_timestep']) \
        .std() \
        .reset_index()

    best_eval_idx = eval_mean_df['reward'].argmax()
    best_eval_run_id = eval_mean_df.iloc[best_eval_idx]['run_id']
    best_eval_eval_no = eval_mean_df.iloc[best_eval_idx]['eval_no']
    best_eval_train_timestep = eval_mean_df.iloc[best_eval_idx]['train_timestep']
    best_eval_reward_value = eval_mean_df.iloc[best_eval_idx]['reward']
    best_eval_reward_std = eval_std_df.iloc[best_eval_idx]['reward']

    print(f"The best evaluation occurred during '{best_eval_run_id}'"
          f'\neval_no={best_eval_eval_no} | train_timestep={best_eval_train_timestep}'
          f'\nepisode_reward={best_eval_reward_value:.2f} +/- {best_eval_reward_std:.2f})')
