import functools
import os
from abc import ABC

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .training_results import TrainingResults

sns.set_palette('colorblind')


def plot_wrapper(
        *,
        fig_id: str,
        fig_title: str,
        xlabel: str,
        ylabel: str,
):
    """
    A decorator for all plots

    Features:
    - Automatically supply ``ax`` parameter to the method as the first parameter
    - Save/show figure based on PlotMaker settings
    - Add labels for axes
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self: 'PlotMakerPlaying', *args, **kwargs):
            fig, ax = plt.subplots(figsize=(12, 5))

            result = method(self, ax, *args, **kwargs)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.tight_layout()

            if self.common_stages_ends is not None:
                for stage_end_step in self.common_stages_ends[:-1]:
                    ax.axvline(
                        x=stage_end_step,
                        linestyle='--',
                        color='black',
                        linewidth=0.5,
                    )

            legend = ax.get_legend()
            if self.is_single_results and legend is not None:
                legend.remove()

            if self.folder:
                fig.savefig(os.path.join(self.folder, f'{fig_id}.png'))
            if self.show:
                ax.set_title(fig_title)
                plt.show()

            return result

        return wrapper

    return decorator


class PlotMaker(ABC):
    """
    All-in-one tool for visualising results of multiple training processes.
    Makes plots shared by playing and passing agents.

    Args:
        results: Training results to visualise. Can be a single object or
            a dictionary with multiple results. In the latter case, keys will be 
            used in plots legends, they can be for example model names.
        folder: A directory where the plots will be saved. Set to ``None``
            to disable saving
        show: Whether or not to call plt.show() for each plot
    """

    def __init__(
            self,
            results: TrainingResults | dict[str, TrainingResults],
            folder: str | None,
            show: bool = False
    ):
        self.folder = folder
        self.show = show

        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)

        if isinstance(results, TrainingResults):
            self.is_single_results = True
            results = {'Model': results}
        else:
            self.is_single_results = False

        training_logs_to_concat = []
        eval_results_to_concat = []
        common_stages_ends = list(results.values())[0].stages_ends

        for model_name, r in results.items():
            training_logs_to_concat.append(
                r.training_logs_df.assign(model=model_name)
            )
            eval_results_to_concat.append(
                r.eval_results_df.assign(model=model_name)
            )
            if r.stages_ends != common_stages_ends:
                common_stages_ends = None

        self._combined_training_logs_df = pd.concat(training_logs_to_concat, ignore_index=True)
        self._combined_eval_results_df = pd.concat(eval_results_to_concat, ignore_index=True)
        self.common_stages_ends = common_stages_ends

    @plot_wrapper(
        fig_id='training_rewards',
        fig_title='Training rewards',
        xlabel='Step',
        ylabel='Reward (rolling mean)',
    )
    def _plot_training_rewards(self, ax: plt.Axes):
        sns.lineplot(
            ax=ax,
            data=self._combined_training_logs_df,
            x='timestep',
            y='ep_rew_mean',
            hue='model',
        )
        ax.grid(axis='y')



class PlotMakerPlaying(PlotMaker):
    """
    All-in-one tool for visualising results of multiple training processes
    of a playing agent
    """

    @plot_wrapper(
        fig_id='eval_points_mean',
        fig_title='Mean points scored in evaluations',
        xlabel='Training timestep',
        ylabel='Mean points scored',
    )
    def _plot_eval_results_mean(self, ax: plt.Axes):
        mean_eval_df = self._combined_eval_results_df \
            .groupby(['model', 'train_timestep']) \
            .mean() \
            .reset_index() \
            .assign(points=lambda df: -df['reward'])
        mean_eval_df = mean_eval_df[['model', 'train_timestep', 'points']]

        sns.lineplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='points',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    @plot_wrapper(
        fig_id='eval_success_rate_mean',
        fig_title='Rounds with fewest points in evaluations',
        xlabel='Training timestep',
        ylabel='% of rounds won',
    )
    def _plot_eval_success_rate_mean(self, ax: plt.Axes):
        mean_eval_df = self._combined_eval_results_df \
            .groupby(['model', 'train_timestep']) \
            .mean() \
            .reset_index()
        mean_eval_df = mean_eval_df[['model', 'train_timestep', 'is_success']]
        mean_eval_df['is_success'] *= 100

        sns.lineplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='is_success',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    @plot_wrapper(
        fig_id='eval_rounds_with_ge13',
        fig_title='Rounds with score >= 13pts in evaluations',
        xlabel='Training timestep',
        ylabel='% of rounds with >=13 pts',
    )
    def _plot_eval_rounds_with_ge13(self, ax: plt.Axes):
        df = self._combined_eval_results_df \
            .assign(points=lambda x: -x['reward']) \
            .assign(is_pts_ge13=lambda x: (x['points'] >= 13) * 100)
        df = df.groupby(['model', 'train_timestep']) \
            .mean() \
            .reset_index()
        sns.lineplot(
            ax=ax,
            data=df,
            x='train_timestep',
            y='is_pts_ge13',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    @plot_wrapper(
        fig_id='eval_moon_shots_success',
        fig_title='Successful moon shots in evaluations',
        xlabel='Training timestep',
        ylabel='# of successful moon shots',
    )
    def _plot_eval_successful_moon_shots(self, ax: plt.Axes):
        df = self._combined_eval_results_df \
            .assign(is_agent_moon_shot=lambda x: x['reward'] == 26)
        df = df.groupby(['model', 'train_timestep']) \
            .sum() \
            .reset_index()
        sns.lineplot(
            ax=ax,
            data=df,
            x='train_timestep',
            y='is_agent_moon_shot',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    @plot_wrapper(
        fig_id='eval_moon_shots_against',
        fig_title='Moon shots against in evaluations',
        xlabel='Training timestep',
        ylabel='# of moon shots against',
    )
    def _plot_eval_moon_shots_against(self, ax: plt.Axes):
        df = self._combined_eval_results_df \
            .assign(is_opponent_moon_shot=lambda x: x['reward'] == -26)
        df = df.groupby(['model', 'train_timestep']) \
            .sum() \
            .reset_index()
        sns.lineplot(
            ax=ax,
            data=df,
            x='train_timestep',
            y='is_opponent_moon_shot',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    def plot_training(self):
        self._plot_training_rewards()

    def plot_eval(self):
        self._plot_eval_results_mean()
        self._plot_eval_success_rate_mean()
        self._plot_eval_rounds_with_ge13()
        self._plot_eval_successful_moon_shots()
        self._plot_eval_moon_shots_against()


class PlotMakerCardPassing(PlotMaker):
    """
    All-in-one tool for visualising results of multiple training processes
    of a card passing agent
    """

    @plot_wrapper(
        fig_id='eval_reward_mean',
        fig_title='Mean reward in evaluations',
        xlabel='Training timestep',
        ylabel='Mean reward',
    )
    def _plot_eval_results_mean(self, ax: plt.Axes):
        mean_eval_df = self._combined_eval_results_df \
            .groupby(['model', 'train_timestep']) \
            .mean() \
            .reset_index()
        mean_eval_df = mean_eval_df[['model', 'train_timestep', 'reward']]

        sns.lineplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='reward',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')

    @plot_wrapper(
        fig_id='eval_success_rate_mean',
        fig_title='Rounds where passing cards reduced points scored',
        xlabel='Training timestep',
        ylabel='Success rate (%)',
    )
    def _plot_eval_success_rate_mean(self, ax: plt.Axes):
        mean_eval_df = self._combined_eval_results_df \
            .groupby(['model', 'train_timestep']) \
            .mean() \
            .reset_index()
        mean_eval_df = mean_eval_df[['model', 'train_timestep', 'is_success']]
        mean_eval_df['is_success'] *= 100

        sns.lineplot(
            ax=ax,
            data=mean_eval_df,
            x='train_timestep',
            y='is_success',
            hue='model',
            marker='o',
        )
        ax.grid(axis='y')
    
    def plot_training(self):
        self._plot_training_rewards()

    def plot_eval(self):
        self._plot_eval_results_mean()
        self._plot_eval_success_rate_mean()
