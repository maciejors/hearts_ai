import numpy as np

from hearts_ai.game import HeartsGame
from hearts_ai.game.players import RandomPlayer, InputPlayer

if __name__ == '__main__':
    players = [
        InputPlayer(),
        RandomPlayer(),
        RandomPlayer(),
        RandomPlayer(),
    ]
    game = HeartsGame(players)
    while np.max(game.core.scoreboard) < 100:
        game.play_round()
