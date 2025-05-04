from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingResults:
    eval_results_df: pd.DataFrame
    all_rewards_df: pd.DataFrame
    training_logs_df: pd.DataFrame
