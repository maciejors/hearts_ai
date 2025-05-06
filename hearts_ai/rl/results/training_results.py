from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingResults:
    eval_results_df: pd.DataFrame
    training_logs_df: pd.DataFrame
    stages_ends: list[int]
