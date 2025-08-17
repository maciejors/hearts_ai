import os
from typing import Literal

import pandas as pd
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from hearts_ai.engine import HeartsRules
from hearts_ai.game import HeartsGame
from hearts_ai.game.players import PPOWrapper, MCTSRLWrapper
from hearts_ai.game.players import RandomPlayer, RuleBasedPlayer, RLPlayer
from hearts_ai.game.players.base import BasePlayer
from hearts_ai.rl.agents import MaskableMCTSRL
from hearts_ai.rl.env import HeartsPlayEnvironment


def ppo(subpath: str) -> PPOWrapper:
    path = os.path.join('output/logs/final_training', subpath)
    agent = MaskablePPO.load(path)
    return PPOWrapper(agent)


def mctsrl(subpath: str) -> MCTSRLWrapper:
    path = os.path.join('output/logs/final_training', subpath)
    env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        reward_setting='sparse',
    )  # env does not matter here, the wrapper will override it
    agent = MaskableMCTSRL.load(path, env=env)
    return MCTSRLWrapper(agent)


def get_mctsrl_player(
        obs_setting_: Literal['full', 'compact'],
        random_state: int
) -> RLPlayer:
    if obs_setting_ == 'full':
        return RLPlayer(
            playing_agent=mctsrl('mctsrl_full/run_1/eval_rule_based/best_model'),
            card_passing_agent=ppo('mctsrl_full/run_1/card_pass/rl_model_222720_steps'),
            play_env_obs_setting=obs_setting_,
            random_state=random_state,
        )
    else:
        return RLPlayer(
            playing_agent=mctsrl('mctsrl_compact/run_1/eval_rule_based/best_model'),
            card_passing_agent=ppo('mctsrl_compact/run_1/card_pass/rl_model_150528_steps'),
            play_env_obs_setting=obs_setting_,
            random_state=random_state,
        )


def get_ppo_player(
        obs_setting_: Literal['full', 'compact'],
        random_state: int
) -> RLPlayer:
    if obs_setting_ == 'full':
        return RLPlayer(
            playing_agent=ppo('ppo_full/run_1/eval_rule_based/best_model'),
            card_passing_agent=ppo('ppo_full/run_1/card_pass/rl_model_87552_steps'),
            play_env_obs_setting=obs_setting_,
            random_state=random_state,
        )
    else:
        return RLPlayer(
            playing_agent=ppo('ppo_compact/run_1/eval_rule_based/best_model'),
            card_passing_agent=ppo('ppo_compact/run_1/card_pass/rl_model_110592_steps'),
            play_env_obs_setting=obs_setting_,
            random_state=random_state,
        )


def run_eval(name: str, players: list[BasePlayer], n_games: int, random_state: int):
    print(f'Now running: {name}')
    results: list[dict] = []
    for game_no in tqdm(range(1, n_games + 1)):
        game = HeartsGame(
            players=players,
            rules=HeartsRules(moon_shot=True, passing_cards=True),
            random_state=random_state,
        )
        while all([score < 100 for score in game.scoreboard]):
            game.pass_cards()
            for trick_no in range(1, 14):
                trick, players_order, winner_idx = game.play_trick()
                for player_idx, card_played in zip(players_order, trick):
                    results.append({
                        'game_no': game_no,
                        'round_no': game.round_no,
                        'trick_no': trick_no,
                        'player_idx': player_idx,
                        'card_played': card_played.idx,
                        'is_trick_winner': player_idx == winner_idx,
                    })
            game.next_round()

    results_df = pd.DataFrame(results)
    save_dir = 'output/final_eval/results'
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(os.path.join(save_dir, f'{name}.csv'), index=False)


if __name__ == '__main__':
    N_GAMES = 1000
    for obs_setting in ['full', 'compact']:
        run_eval(
            name=f'ppo_{obs_setting}_vs_random',
            players=[
                get_ppo_player(obs_setting, random_state=0),  #type: ignore
                RandomPlayer(1), 
                RandomPlayer(2), 
                RandomPlayer(3),
            ],
            n_games=N_GAMES,
            random_state=28,
        )
        run_eval(
            name=f'ppo_{obs_setting}_vs_rule_based',
            players=[
                get_ppo_player(obs_setting, random_state=0),  #type: ignore
                RuleBasedPlayer(), 
                RuleBasedPlayer(), 
                RuleBasedPlayer(),
            ],
            n_games=N_GAMES,
            random_state=28,
        )
        run_eval(
            name=f'mctsrl_{obs_setting}_vs_random',
            players=[
                get_mctsrl_player(obs_setting, random_state=0),  #type: ignore
                RandomPlayer(1), 
                RandomPlayer(2), 
                RandomPlayer(3)
            ],
            n_games=N_GAMES,
            random_state=28,
        )
        run_eval(
            name=f'mctsrl_{obs_setting}_vs_rule_based',
            players=[
                get_mctsrl_player(obs_setting, random_state=0),  #type: ignore
                RuleBasedPlayer(), 
                RuleBasedPlayer(), 
                RuleBasedPlayer()
            ],
            n_games=N_GAMES,
            random_state=28,
        )
