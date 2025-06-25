import copy

import numpy as np

from .constants import (
    MAX_POINTS, PLAYER_COUNT, CARDS_PER_PLAYER_COUNT, PassDirection, CARDS_TO_PASS_COUNT,
)
from .deck import Deck, Card, Suit
from .rules import HeartsRules
from .utils import is_heart, points_for_card, is_starting_card, get_trick_winner_idx


def _get_empty_cards_list_per_player() -> list[list[Card]]:
    return [[] for _ in range(PLAYER_COUNT)]


class HeartsRound:
    """
    Engine for the standard 4-player game of Hearts

    Args:
        rules: Toggleable rules of the engine. Set to ``None`` for default
            rules (see :class:`HeartsRules` for defaults)
        pass_direction: Pass direction for this round. This parameter is
            ignored if card passing rule is disabled - in such case,
            pass direction will always be NO_PASSING. Default is ``None``,
            which means it will be set to LEFT if the card passing rule is
            enabled.
        random_state: Random seed for reproducibility
    """

    __slots__ = [
        '_rng',
        'rules',
        '_pass_direction',
        '_hands',
        '_taken_cards',
        '_played_cards',
        'trick_no',
        'are_hearts_broken',
        '_trick_starting_player_idx',
        '_current_trick',
        '__cards_to_pass',
        'are_cards_passed',
    ]

    def __init__(self,
                 rules: HeartsRules = HeartsRules(),
                 pass_direction: PassDirection | None = None,
                 random_state: int | None = None):

        self._rng = np.random.default_rng(random_state)
        self.rules = rules

        if not self.rules.passing_cards:
            self._pass_direction = PassDirection.NO_PASSING
        elif pass_direction is not None:
            self._pass_direction = pass_direction
        else:
            self._pass_direction = PassDirection.LEFT

        deck = Deck(random_state=random_state)
        deck.shuffle()

        self._hands = _get_empty_cards_list_per_player()
        self._taken_cards = _get_empty_cards_list_per_player()
        self._played_cards = _get_empty_cards_list_per_player()

        for player_idx in range(PLAYER_COUNT):
            self._hands[player_idx] = deck.deal(CARDS_PER_PLAYER_COUNT).tolist()

        self.trick_no = 1
        self.are_hearts_broken = False

        self._trick_starting_player_idx: int | None = None
        self._current_trick: list[Card] = []

        self.__cards_to_pass = _get_empty_cards_list_per_player()

        self.are_cards_passed = False
        if not self.__can_perform_card_passing():
            self.are_cards_passed = True

        self.set_starting_player()

    @property
    def pass_direction(self) -> PassDirection:
        return self._pass_direction

    @pass_direction.setter
    def pass_direction(self, value):
        self._pass_direction = value

    @property
    def hands(self) -> list[list[Card]]:
        return [c.copy() for c in self._hands]

    @hands.setter
    def hands(self, value):
        self._hands = value
        # this is particularly useful for pass env
        self.set_starting_player()

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
    def points_collected(self) -> list[int]:
        """
        The number of points collected by each player in the round.
        Does not take the moon shot into account.
        """
        return [sum(points_for_card(card) for card in self._taken_cards[i])
                for i in range(PLAYER_COUNT)]

    @property
    def scores(self) -> list[int]:
        """
        The score of each player in the round.
        Takes the moon shot into account.
        """
        round_scores = self.points_collected.copy()

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
        return MAX_POINTS in self.points_collected

    @property
    def is_finished(self) -> bool:
        return self.trick_no == CARDS_PER_PLAYER_COUNT + 1

    def set_starting_player(self):
        """
        Set the starting player to the player with 2 of clubs on hand
        """
        for player_idx in range(PLAYER_COUNT):
            if any(is_starting_card(card) for card in self._hands[player_idx]):
                self._trick_starting_player_idx = player_idx
                return

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
            raise RuntimeError('Cannot play card because the trick is full. '
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
        winner_idx = get_trick_winner_idx(
            current_trick,
            trick_starting_player_idx=self.trick_starting_player_idx,
        )

        self._taken_cards[winner_idx].extend(current_trick)
        self._trick_starting_player_idx = winner_idx
        self._current_trick = []
        self.trick_no += 1

        return current_trick, winner_idx

    def next(self) -> 'HeartsRound':
        """
        Get the next round object, with the same rules.
        This will return a fresh round object, but with updated pass direction
        is this rule is enabled.
        """
        if not self.rules.passing_cards:
            next_pass_direction = PassDirection.NO_PASSING
        else:
            all_pass_directions = list(PassDirection)
            next_pass_direction = all_pass_directions[(self.pass_direction.value + 1) % len(all_pass_directions)]

        next_round = HeartsRound(
            rules=self.rules,
            pass_direction=next_pass_direction,
            random_state=int(self._rng.integers(999999))
        )
        return next_round

    def __deepcopy__(self, memo):
        """
        Note:
            only things that need to be deep-copied for the environments are copied.
        """
        round_copy = copy.copy(self)
        round_copy._hands = [card_list.copy() for card_list in self._hands]
        round_copy._played_cards = [card_list.copy() for card_list in self._played_cards]
        round_copy._taken_cards = [card_list.copy() for card_list in self._taken_cards]
        round_copy._current_trick = self._current_trick.copy()
        return round_copy
