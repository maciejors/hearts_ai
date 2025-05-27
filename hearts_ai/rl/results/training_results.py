from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainingResults:
    """Results of a single training process"""
    eval_results: dict[str, pd.DataFrame]
    training_logs: pd.DataFrame
    stages_ends: list[int]
