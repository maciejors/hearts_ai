import os

import matplotlib.pyplot as plt
import seaborn as sns

from .training_results import TrainingResults


class PlotMakerIndividualResults:
    """
    All-in-one tool for visualising results of a single training process

    Args:
        results: Training results to visualise
        folder: A directory where the plots will be saved. Set to ``None``
            to disable saving
        show: Whether or not to call plt.show() for each plot
    """

    def __init__(
            self,
            results: TrainingResults,
            folder: str | None,
            show: bool = False
    ):
        self._r = results
        self._folder = folder
        self._show = show

        if self._folder is not None:
            os.makedirs(self._folder, exist_ok=True)

    @staticmethod
    def __get_default_subplots() -> tuple[plt.Figure, plt.Axes]:
        return plt.subplots(figsize=(15, 5))

    def __handle_ready_plot(
            self,
            fig: plt.Figure,
            ax: plt.Axes,
            fig_id: str,
            fig_title: str,
    ):
        fig.tight_layout()

        if self._folder:
            fig.savefig(os.path.join(self._folder, f'{fig_id}.png'))

        if self._show:
            ax.set_title(fig_title)
            plt.show()

    def _plot_training_rewards(self):
        fig, ax = self.__get_default_subplots()
        sns.lineplot(
            ax=ax,
            data=self._r.training_logs_df,
            x='timestep',
            y='ep_rew_mean',
            hue='stage'
        )
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward (rolling mean)')
        self.__handle_ready_plot(
            fig, ax, 'training_rewards', 'Training rewards'
        )

    def _plot_evaluation_results_boxplot(self):
        fig, ax = self.__get_default_subplots()
        sns.boxplot(
            ax=ax,
            data=self._r.eval_results_df,
            x='train_timestep',
            y='reward',
        )
        ax.set_xlabel('Training timestep')
        ax.set_ylabel('Evaluation rewards')
        self.__handle_ready_plot(
            fig, ax, 'evaluation_rewards_boxplot', 'Evaluation rewards'
        )

    def _plot_evaluation_results_mean_line(self):
        mean_eval_df = self._r.eval_results_df \
            .groupby('train_timestep') \
            .mean() \
            .reset_index()
        mean_eval_df = mean_eval_df[['train_timestep', 'reward']]

        fig, ax = self.__get_default_subplots()
        sns.lineplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='reward',
        )
        sns.scatterplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='reward',
            s=50,
            marker='o'
        )
        ax.set_xlabel('Training timestep')
        ax.set_ylabel('Mean evaluation reward')
        self.__handle_ready_plot(
            fig, ax, 'evaluation_rewards_mean_line', 'Evaluation rewards'
        )

    def plot_all(self):
        self._plot_training_rewards()
        self._plot_evaluation_results_boxplot()
        self._plot_evaluation_results_mean_line()
