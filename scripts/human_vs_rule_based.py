import numpy as np

from hearts_ai.game import HeartsGame
from hearts_ai.game.players import RuleBasedPlayer, InputPlayer

if __name__ == '__main__':
    players = [
        InputPlayer(),
        RuleBasedPlayer(),
        RuleBasedPlayer(),
        RuleBasedPlayer(),
    ]
    game = HeartsGame(players)
    while np.max(game.round.scoreboard) < 100:
        game.play_round()
        game.next_round()
        print(f'Current scores (yours is first): {game.scoreboard}')
