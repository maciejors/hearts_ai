import os

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


def mctsrl(subpath: str, player_idx: int) -> MCTSRLWrapper:
    path = os.path.join('output/logs/final_training', subpath)
    env = HeartsPlayEnvironment(
        opponents_callbacks=[],
        reward_setting='sparse',
    )  # env does not matter here, the wrapper will override it
    agent = MaskableMCTSRL.load(path, env=env)
    return MCTSRLWrapper(agent, player_idx)


def get_mctsrl_player(player_idx: int, random_state: int) -> RLPlayer:
    return RLPlayer(
        playing_agent=mctsrl('mctsrl_play_ppo_pass/run_1/eval_rule_based/best_model', player_idx),
        card_passing_agent=ppo('mctsrl_play_ppo_pass/run_1/card_pass/rl_model_222720_steps'),
        random_state=random_state,
    )


def get_ppo_player(random_state: int) -> RLPlayer:
    return RLPlayer(
        playing_agent=ppo('ppo_both/run_1/eval_rule_based/best_model'),
        card_passing_agent=ppo('ppo_both/run_1/card_pass/rl_model_87552_steps'),
        random_state=random_state,
    )


def run_eval(name: str, players: list[BasePlayer], n_games: int, random_state: int):
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
    results_df.to_csv(f'output/final_eval/{name}.csv', index=False)


if __name__ == '__main__':
    N_GAMES = 1000
    run_eval(
        name='ppo_vs_random',
        players=[get_ppo_player(0), get_ppo_player(1), RandomPlayer(2), RandomPlayer(3)],
        n_games=N_GAMES,
        random_state=28,
    )
    run_eval(
        name='ppo_vs_rule_based',
        players=[get_ppo_player(0), get_ppo_player(1), RuleBasedPlayer(), RuleBasedPlayer()],
        n_games=N_GAMES,
        random_state=28,
    )
    run_eval(
        name='mctsrl_vs_random',
        players=[get_mctsrl_player(0, 0), get_mctsrl_player(1, 1), RandomPlayer(2), RandomPlayer(3)],
        n_games=N_GAMES,
        random_state=28,
    )
    run_eval(
        name='mctsrl_vs_rule_based',
        players=[get_mctsrl_player(0, 0), get_mctsrl_player(1, 1), RuleBasedPlayer(), RuleBasedPlayer()],
        n_games=N_GAMES,
        random_state=28,
    )
    run_eval(
        name='tournament',
        players=[get_ppo_player(8), get_mctsrl_player(1, 26), RuleBasedPlayer(), RandomPlayer(4)],
        n_games=N_GAMES,
        random_state=24,
    )
