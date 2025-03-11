from game import HeartsGame
from players import RandomPlayer, InputPlayer

if __name__ == '__main__':
    players = [
        InputPlayer(),
        RandomPlayer(random_state=11),
        RandomPlayer(random_state=22),
        RandomPlayer(random_state=33),
    ]
    game = HeartsGame(players, random_state=44)
    game.play_round()
