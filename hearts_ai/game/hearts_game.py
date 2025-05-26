from hearts_ai.engine import HeartsRound, HeartsRules
from hearts_ai.engine.constants import (
    PLAYER_COUNT, CARDS_PER_PLAYER_COUNT,
)
from .players.base import BasePlayer


class HeartsGame:
    """
    Args:
        players: List of exactly 4 player 'brains'
        rules: Toggleable rules of the engine. Set to ``None`` for default
            rules (see :class:`HeartsGameRules` for defaults)
        random_state: Random seed for reproducibility. Does not control the
            randomness of players.
    """

    def __init__(self,
                 players: list[BasePlayer],
                 rules: HeartsRules = HeartsRules(),
                 random_state: int | None = None):

        if len(players) != PLAYER_COUNT:
            raise ValueError(f'There should be exactly {PLAYER_COUNT} players')

        self.players = players
        self.round = HeartsRound(
            rules=rules,
            random_state=random_state,
        )
        self.scoreboard = [0 for _ in range(PLAYER_COUNT)]

    def pass_cards(self):
        if self.round.are_cards_passed:
            return

        for player_idx, player in enumerate(self.players):
            selected_cards = player.select_cards_to_pass(
                self.round.hands[player_idx], self.round.pass_direction
            )
            self.round.pick_cards_to_pass(player_idx, selected_cards)
        self.round.complete_pass_cards()

    def play_trick(self):
        for _ in range(PLAYER_COUNT):
            current_player_idx = self.round.current_player_idx
            player = self.players[current_player_idx]
            card = player.play_card(
                hand=self.round.hands[current_player_idx],
                trick=self.round.current_trick,
                are_hearts_broken=self.round.are_hearts_broken,
                is_first_trick=self.round.trick_no == 1,
            )
            self.round.play_card(card)

        trick, winner_idx = self.round.complete_trick()
        for i, player in enumerate(self.players):
            player.post_trick_callback(trick, i == winner_idx)

    def play_round(self):
        self.pass_cards()
        for trick_num in range(CARDS_PER_PLAYER_COUNT):
            self.play_trick()

        for player, score in zip(self.players, self.round.scores):
            player.post_round_callback(score)

    def next_round(self):
        for player_idx, player_score in enumerate(self.round.scores):
            self.scoreboard[player_idx] += player_score
        self.round = self.round.next()
