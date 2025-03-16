from dataclasses import dataclass

from hearts_ai.constants import (
    MAX_POINTS, PLAYER_COUNT, CARDS_PER_PLAYER, PassDirection,
)
from hearts_ai.players.base import BasePlayer
from hearts_ai.utils import is_heart, points_for_card, is_starting_card
from .deck import Deck, Card


@dataclass
class HeartsGameRules:
    """
    Args:
        moon_shot: Determines whether shooting the moon is enabled
        passing_cards: Determines whether passing cards is enabled
    """
    moon_shot: bool = True
    passing_cards: bool = True


class HeartsGame:
    """
    Args:
        players: List of exactly 4 player 'brains'
        rules: Toggleable rules of the game. Set to ``None`` for default
            rules (see :class:`HeartsGameRules` for defaults)
        random_state: Random seed for reproducibility
    """

    def __init__(self,
                 players: list[BasePlayer],
                 rules: HeartsGameRules = HeartsGameRules(),
                 random_state: int | None = None):

        if len(players) != PLAYER_COUNT:
            raise ValueError(f'There should be exactly {PLAYER_COUNT} players')

        self.players = players
        self.deck = Deck(random_state=random_state)
        self.rules = rules

        self.round_no = 0
        self.trick_no = 0
        self.are_hearts_broken = False
        # order of items in the list corresponds to the order of players
        self.hands = [[] for _ in self.players]
        self.taken_cards = [[] for _ in self.players]
        self.starting_player_idx = None

        # total points excluding points from the current round
        self.scoreboard = [0 for _ in self.players]
        self.pass_direction_int = PassDirection.LEFT.value

    @property
    def current_round_points(self) -> list[int]:
        round_scores = [0 for _ in self.players]

        for i, _ in enumerate(self.players):
            for card in self.taken_cards[i]:
                round_scores[i] += points_for_card(card)

        if self.rules.moon_shot and MAX_POINTS in round_scores:
            shooter_idx = round_scores.index(MAX_POINTS)
            for i, _ in enumerate(self.players):
                if i != shooter_idx:
                    self.scoreboard[i] += MAX_POINTS
                else:
                    self.scoreboard[i] -= MAX_POINTS
        return round_scores

    def set_starting_player(self):
        """Set the starting player to the player with 2 of clubs on hand"""
        for player_idx, _ in enumerate(self.players):
            if any(is_starting_card(card) for card in self.hands[player_idx]):
                self.starting_player_idx = player_idx
                return

    def pre_round(self):
        """
        Ends the current round (if one has been played) and
        prepares the state for the next round.
        """
        if self.round_no > 0:
            self.pass_direction_int = (self.pass_direction_int + 1) % 4

        # prepare the next round
        self.round_no += 1
        self.trick_no = 0
        self.deck.shuffle()
        self.are_hearts_broken = False
        for player_idx, _ in enumerate(self.players):
            self.hands[player_idx] = list(self.deck.deal(CARDS_PER_PLAYER))
            self.taken_cards[player_idx] = []

        self.set_starting_player()

    def post_round(self):
        # update the scoreboard
        round_scores = self.current_round_points
        for i, player in enumerate(self.players):
            score_for_player = round_scores[i]
            self.scoreboard[i] += score_for_player
            player.post_round_callback(score_for_player)

    def pass_cards(self):
        if (self.pass_direction_int == PassDirection.NO_PASSING.value
                or not self.rules.passing_cards):
            return

        pass_offsets = [1, 3, 2]
        pass_offset = pass_offsets[self.pass_direction_int]

        # 1. select the cards to pass
        cards_to_pass = []
        for i, player in enumerate(self.players):
            selected_cards = player.select_cards_to_pass(self.hands[i], self.pass_direction_int)

            for card in selected_cards:
                self.hands[i].remove(card)

            cards_to_pass.append(selected_cards)

        # 2. pass the cards
        for i, _ in enumerate(self.players):
            target_idx = (i + pass_offset) % PLAYER_COUNT
            self.hands[target_idx].extend(cards_to_pass[i])
        # just in case 2 of clubs was passed
        self.set_starting_player()

    def play_trick(self) -> tuple[list[Card], int]:
        """
        Returns:
            A tuple of two elements, the first one is the trick content, and
            the second is the index of a player who took the trick.
        """
        if self.trick_no == 13:
            raise RuntimeError('The round has ended. The trick cannot be played')

        self.trick_no += 1
        current_trick = []
        current_player_idx = self.starting_player_idx

        for _ in range(PLAYER_COUNT):
            player = self.players[current_player_idx]

            card = player.play_card(
                hand=self.hands[current_player_idx],
                trick=current_trick.copy(),
                are_hearts_broken=self.are_hearts_broken,
                is_first_trick=self.trick_no == 1,
            )
            self.hands[current_player_idx].remove(card)
            current_trick.append(card)

            if is_heart(card) and not self.are_hearts_broken:
                self.are_hearts_broken = True

            current_player_idx = (current_player_idx + 1) % PLAYER_COUNT

        # check who is taking the trick
        lead_suit = current_trick[0].suit
        winning_card = current_trick[0]
        winner_idx = self.starting_player_idx

        for i in range(1, PLAYER_COUNT):
            player_idx = (self.starting_player_idx + i) % PLAYER_COUNT
            card = current_trick[i]

            if card.suit == lead_suit and card.rank_value > winning_card.rank_value:
                winning_card = card
                winner_idx = player_idx

        self.taken_cards[winner_idx].extend(current_trick)
        self.starting_player_idx = winner_idx

        for i, player in enumerate(self.players):
            player.post_trick_callback(current_trick, i == winner_idx)

        return current_trick, winner_idx

    def play_round(self):
        self.pre_round()

        self.pass_cards()
        for trick_num in range(13):
            self.play_trick()

        self.post_round()
