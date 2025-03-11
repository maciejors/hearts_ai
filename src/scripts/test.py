from game import HeartsGame
from players import RandomPlayer, InputPlayer

if __name__ == '__main__':
    players = [
        InputPlayer(),
        RandomPlayer(),
        RandomPlayer(),
        RandomPlayer(),
    ]
    game = HeartsGame(players)
    game.play_round()
