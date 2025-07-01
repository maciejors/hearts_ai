import numpy as np

from hearts_ai.engine import HeartsRound, HeartsRules, Card
from hearts_ai.engine.constants import (
    PLAYER_COUNT, CARDS_PER_PLAYER_COUNT,
)
from hearts_ai.engine.round import STATE_IDX_TAKEN_CARDS, STATE_IDX_HANDS_INFO_START
from .players.base import BasePlayer
from .utils import array_to_card_list, card_list_to_array


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

    @property
    def hands(self) -> list[list[Card]]:
        return [
            array_to_card_list(self.round.get_hand(player_idx))
            for player_idx in range(PLAYER_COUNT)
        ]

    @property
    def taken_cards(self) -> list[list[Card]]:
        taken_cards_info = self.round.get_numpy_state()[STATE_IDX_TAKEN_CARDS]
        result = []
        for player_idx in range(PLAYER_COUNT):
            taken_cards_for_player = np.where(taken_cards_info == player_idx)[0]
            result.append(array_to_card_list(taken_cards_for_player))
        return result

    @property
    def played_cards(self) -> list[list[Card]]:
        result = []
        round_np_state = self.round.get_numpy_state()
        for idx in STATE_IDX_HANDS_INFO_START:
            played_cards_for_player = np.where(round_np_state[idx:idx + 52] == -1)[0]
            result.append(array_to_card_list(played_cards_for_player))
        return result

    @property
    def round_scores(self) -> list[int]:
        return self.round.scores.tolist()

    def pass_cards(self):
        if self.round.are_cards_passed:
            return

        for player_idx, player in enumerate(self.players):
            selected_cards = player.select_cards_to_pass(
                self.hands[player_idx],
                self.round.pass_direction,
            )
            self.round.pick_cards_to_pass(player_idx, card_list_to_array(selected_cards))
        self.round.perform_cards_passing()

    def play_trick(self):
        for _ in range(PLAYER_COUNT):
            current_player_idx = self.round.current_player_idx
            player = self.players[current_player_idx]
            card = player.play_card(
                hand=self.hands[current_player_idx],
                trick=array_to_card_list(self.round.current_trick_ordered),
                are_hearts_broken=self.round.are_hearts_broken,
                is_first_trick=self.round.trick_no == 1,
            )
            self.round.play_card(card.idx)

        trick, winner_idx = self.round.complete_trick(return_ordered=True)
        for i, player in enumerate(self.players):
            player.post_trick_callback(array_to_card_list(trick), i == winner_idx)

    def play_round(self):
        self.pass_cards()
        for _ in range(CARDS_PER_PLAYER_COUNT):
            self.play_trick()

        for player, score in zip(self.players, self.round_scores):
            player.post_round_callback(score)

    def next_round(self):
        for player_idx, player_score in enumerate(self.round_scores):
            self.scoreboard[player_idx] += player_score
        self.round = self.round.next()
