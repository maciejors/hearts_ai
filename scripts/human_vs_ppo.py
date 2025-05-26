import os.path

import numpy as np
from sb3_contrib import MaskablePPO

from hearts_ai.game import HeartsGame
from hearts_ai.game.players import InputPlayer, RLPlayer

model_play_path = os.path.abspath(
    './output/logs/ppo_playing_compare_stages_setups/every_192x1000/eval/best_model'
)
model_card_pass_path = os.path.abspath(
    './output/logs/ppo_passing_compare_stages_setups/every_512x50/eval/best_model'
)

if __name__ == '__main__':
    opponents = [
        RLPlayer(
            playing_agent=MaskablePPO.load(model_play_path),
            card_passing_agent=MaskablePPO.load(model_card_pass_path),
        )
        for _ in range(3)
    ]
    game = HeartsGame([InputPlayer(), *opponents])
    while np.max(game.round.scoreboard) < 100:
        game.play_round()
        game.next_round()

    print(f'Game finished')
    print(f'Your score: {game.round.scoreboard[0]}')
    print(f'All scores: {game.round.scoreboard}')
