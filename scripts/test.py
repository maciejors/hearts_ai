from game import HeartsGame
from players import RandomPlayer, InputPlayer

if __name__ == '__main__':
    players = [
        InputPlayer(),
        RandomPlayer(random_state=24),
        RandomPlayer(random_state=28),
        RandomPlayer(random_state=3),
    ]
    game = HeartsGame(players, random_state=24)
    game.play_round()
    print(game.current_round_points)
    print(game.scoreboard)
