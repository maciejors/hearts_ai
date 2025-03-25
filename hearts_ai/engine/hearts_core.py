from dataclasses import dataclass

from .constants import (
    MAX_POINTS, PLAYER_COUNT, CARDS_PER_PLAYER_COUNT, PassDirection, CARDS_TO_PASS_COUNT,
)
from .deck import Deck, Card, Suit
from .utils import is_heart, points_for_card, is_starting_card


@dataclass
class HeartsRules:
    """
    Args:
        moon_shot: Determines whether shooting the moon is enabled
        passing_cards: Determines whether passing cards is enabled
    """
    moon_shot: bool = True
    passing_cards: bool = True


class HeartsCore:
    """
    Engine for the standard 4-player game of Hearts

    Args:
        rules: Toggleable rules of the engine. Set to ``None`` for default
            rules (see :class:`HeartsRules` for defaults)
        random_state: Random seed for reproducibility
    """

    def __init__(self,
                 rules: HeartsRules = HeartsRules(),
                 random_state: int | None = None):

        self.deck = Deck(random_state=random_state)
        self.rules = rules

        self.round_no = 0
        self.trick_no = 0
        self.are_hearts_broken = False

        self._hands: list[list[Card]] = [[] for _ in range(PLAYER_COUNT)]
        self._taken_cards: list[list[Card]] = [[] for _ in range(PLAYER_COUNT)]
        self._played_cards: list[list[Card]] = [[] for _ in range(PLAYER_COUNT)]
        self._trick_starting_player_idx: int | None = None
        self._current_trick: list[Card] = []

        self.__cards_to_pass: list[list[Card]] = [[] for _ in range(PLAYER_COUNT)]
        self.are_cards_passed = False

        self.scoreboard = [0 for _ in range(PLAYER_COUNT)]

    @property
    def hands(self) -> list[list[Card]]:
        return [c.copy() for c in self._hands]

    @property
    def taken_cards(self) -> list[list[Card]]:
        return [c.copy() for c in self._taken_cards]

    @property
    def played_cards(self) -> list[list[Card]]:
        return [c.copy() for c in self._played_cards]

    @property
    def trick_starting_player_idx(self) -> int | None:
        return self._trick_starting_player_idx

    @property
    def current_trick(self) -> list[Card]:
        return self._current_trick.copy()

    @property
    def pass_direction(self) -> PassDirection:
        pass_directions_ordered = list(PassDirection)
        return pass_directions_ordered[max(self.round_no - 1, 0) % len(pass_directions_ordered)]

    @property
    def current_round_points_collected(self) -> list[int]:
        """
        The number of points collected by each player in the round.
        Does not take the moon shot into account.
        """
        return [sum(points_for_card(card) for card in self._taken_cards[i])
                for i in range(PLAYER_COUNT)]

    @property
    def current_round_scores(self) -> list[int]:
        """
        The score of each player in the round.
        Takes the moon shot into account.
        """
        round_scores = self.current_round_points_collected.copy()

        if self.rules.moon_shot and self.is_moon_shot_triggered:
            shooter_idx = round_scores.index(MAX_POINTS)
            for player_idx in range(PLAYER_COUNT):
                if player_idx != shooter_idx:
                    round_scores[player_idx] += MAX_POINTS
                else:
                    round_scores[player_idx] -= MAX_POINTS
        return round_scores

    @property
    def current_player_idx(self) -> int:
        """ID of the player that is expected to throw the next card"""
        return (self._trick_starting_player_idx + len(self._current_trick)) % PLAYER_COUNT

    @property
    def leading_suit(self) -> Suit | None:
        """Leading suit in the current trick, or None if the trick is empty"""
        if len(self.current_trick) == 0:
            return None
        return self.current_trick[0].suit

    @property
    def is_current_trick_full(self) -> bool:
        return len(self._current_trick) == PLAYER_COUNT

    @property
    def is_moon_shot_triggered(self) -> bool:
        """Check if any of the players have shot the moon in this round"""
        return MAX_POINTS in self.current_round_points_collected

    @property
    def is_round_finished(self) -> bool:
        return self.trick_no == CARDS_PER_PLAYER_COUNT + 1

    def set_starting_player(self):
        """
        Set the starting player to the player with 2 of clubs on hand
        """
        for player_idx in range(PLAYER_COUNT):
            if any(is_starting_card(card) for card in self._hands[player_idx]):
                self._trick_starting_player_idx = player_idx
                return

    def next_round(self):
        """
        Completes the current round and updates the state for the next round.
        """
        for player_idx in range(PLAYER_COUNT):
            self.scoreboard[player_idx] += self.current_round_scores[player_idx]

        # prepare the next round
        self.round_no += 1
        self.trick_no = 1
        self.deck.shuffle()
        self.are_hearts_broken = False
        for player_idx in range(PLAYER_COUNT):
            self._hands[player_idx] = list(self.deck.deal(CARDS_PER_PLAYER_COUNT))
            self._taken_cards[player_idx] = []
            self._played_cards[player_idx] = []

        self.set_starting_player()

        self.__cards_to_pass = [[] for _ in range(PLAYER_COUNT)]
        self.are_cards_passed = False
        if self.pass_direction == PassDirection.NO_PASSING or not self.rules.passing_cards:
            self.are_cards_passed = True

    def __can_perform_card_passing(self) -> bool:
        """
        Returns ``True`` if a passing cards can be performed
        """
        if self.pass_direction == PassDirection.NO_PASSING:
            return False
        if not self.rules.passing_cards:
            return False
        return not self.are_cards_passed

    def pick_cards_to_pass(self, player_idx: int, cards: list[Card]):
        if not self.__can_perform_card_passing():
            return
        self.__cards_to_pass[player_idx] = cards
        for card in cards:
            self._hands[player_idx].remove(card)

    def complete_pass_cards(self):
        if not self.__can_perform_card_passing():
            return

        if any([len(cards) != CARDS_TO_PASS_COUNT for cards in self.__cards_to_pass]):
            raise ValueError('Cannot pass the cards if not all players selected '
                             f'their {CARDS_TO_PASS_COUNT} cards to pass.')

        pass_offset = [1, 3, 2][self.pass_direction.value]

        for player_idx in range(PLAYER_COUNT):
            target_idx = (player_idx + pass_offset) % PLAYER_COUNT
            self._hands[target_idx].extend(self.__cards_to_pass[player_idx])
            self.__cards_to_pass[player_idx] = []

        self.are_cards_passed = True
        # just in case 2 of clubs was passed
        self.set_starting_player()

    def play_card(self, card: Card):
        """
        Play a card in the current trick
        """
        if self.trick_no > CARDS_PER_PLAYER_COUNT:
            raise RuntimeError('The round has ended. The trick cannot be played')
        if self.is_current_trick_full:
            raise RuntimeError('Cannnot play card because the trick is full. '
                               'Complete the trick before playing the next card')
        if not self.are_cards_passed:
            raise RuntimeError('Cannot play any cards before cards are passed.')

        self._hands[self.current_player_idx].remove(card)
        self._played_cards[self.current_player_idx].append(card)
        self._current_trick.append(card)

        if is_heart(card) and not self.are_hearts_broken:
            self.are_hearts_broken = True

    def complete_trick(self) -> tuple[list[Card], int]:
        """
        Complete the current trick and prepare for the next one

        Returns:
            A tuple of two elements, the first one is the trick content, and
            the second is the index of a player who took the trick.
        """
        current_trick = self.current_trick

        winning_card = current_trick[0]
        winner_idx = self.trick_starting_player_idx

        for i in range(1, PLAYER_COUNT):
            player_idx = (self.trick_starting_player_idx + i) % PLAYER_COUNT
            card = current_trick[i]

            if card.suit == self.leading_suit and card.rank_value > winning_card.rank_value:
                winning_card = card
                winner_idx = player_idx

        self._taken_cards[winner_idx].extend(current_trick)
        self._trick_starting_player_idx = winner_idx
        self._current_trick = []
        self.trick_no += 1

        return current_trick, winner_idx
