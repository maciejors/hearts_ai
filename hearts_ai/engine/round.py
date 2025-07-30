import numpy as np

from .constants import (
    MAX_POINTS, PLAYER_COUNT, CARDS_PER_PLAYER_COUNT, PassDirection, CARDS_TO_PASS_COUNT,
)
from .rules import HeartsRules
from .utils import is_heart, points_for_card, get_winning_card_argmax

STATE_IDX_TRICK_NO = 0
STATE_IDX_HANDS_INFO_START = [1, 53, 105, 157]
STATE_IDX_LEAD_SUITS = [209, 210, 211, 212]
STATE_IDX_POINTS_COLLECTED = [213, 214, 215, 216]
STATE_IDX_ARE_HEARTS_BROKEN = 217
STATE_IDX_TRICK_STARTING_PLAYER = 218
STATE_IDX_TAKEN_CARDS = np.arange(219, 271)


class HeartsRound:
    """
    Engine for the standard 4-player game of Hearts

    The whole state of the game can be obtained through a method
    ``get_numpy_state()`` which returns the state from the perspective of the
    current player (i.e. the player that has to choose their card to play now).
    This is ``None`` during card passing phase. Each state is a vector of
    length 261, containing the following information (by indices):

        0: Trick number
        1-52: Player 0's relation to each card
            (-1: played before, 0: not in hand, 1: in hand, 2: played in this trick)
        53-104: Player 1's relation to each card
        105-156: Player 2's relation to each card
        157-208: Player 3's relation to each card
        209-212: One-hot vector specifying the leading suit in this trick
        213-216: Points collected so far in the game by each player
        217: Are hearts broken? (0: no, 1: yes)
        218: Index of a player starting this trick (or 4 if cards are not passed yet)
        219-270: For each card, an index of a player who has taken it
            (-1: not taken by anyone, 0-3: index of a player who has taken it)

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

    def __init__(self,
                 rules: HeartsRules = HeartsRules(),
                 pass_direction: PassDirection | None = None,
                 random_state: int | np.int_ | None = None):

        rng = np.random.default_rng(random_state)
        self._next_seed = lambda: rng.integers(999999)

        self.rules = rules

        if not self.rules.passing_cards:
            self._pass_direction = PassDirection.NO_PASSING
        elif pass_direction is not None:
            self._pass_direction = pass_direction
        else:
            self._pass_direction = PassDirection.LEFT

        self._np_state = np.zeros(271, dtype=np.int8)
        self._np_state[STATE_IDX_TRICK_NO] = 1
        self._np_state[STATE_IDX_TAKEN_CARDS] = -1

        deck = np.arange(52)
        rng.shuffle(deck)
        for player_idx in range(PLAYER_COUNT):
            hand = deck[player_idx * 13:(player_idx + 1) * 13]
            self._np_state[STATE_IDX_HANDS_INFO_START[player_idx] + hand] = 1

        self.__cards_to_pass: list[np.ndarray] = [np.array([], dtype=np.int8) for _ in range(PLAYER_COUNT)]
        self.are_cards_passed = False
        if not self.__can_perform_card_passing():
            self.are_cards_passed = True

        self.set_starting_player()

    def get_numpy_state(self) -> np.ndarray:
        return self._np_state.copy()

    @property
    def pass_direction(self) -> PassDirection:
        return self._pass_direction

    @pass_direction.setter
    def pass_direction(self, value):
        self._pass_direction = value

    @property
    def trick_no(self) -> np.int_:
        return self._np_state[STATE_IDX_TRICK_NO]  # type: ignore

    @property
    def are_hearts_broken(self) -> bool:
        return self._np_state[STATE_IDX_ARE_HEARTS_BROKEN] == 1

    @property
    def trick_starting_player_idx(self) -> np.int_ | None:
        if not self.are_cards_passed:
            return None
        return self._np_state[STATE_IDX_TRICK_STARTING_PLAYER]

    @property
    def current_trick_unordered(self) -> np.ndarray:
        """
        Returns:
            List of card indexes in the current trick. The order of cards
            is based on the order of players indexes. This is
            computationally cheaper than ``current_trick_ordered``
        """
        return np.where(self._np_state[1:209] == 2)[0] % 52

    @property
    def current_trick_ordered(self) -> np.ndarray:
        """
        Returns:
            List of card indexes in the current trick. The order of cards
            is the order they were played (i.e. the first is the lead card)
        """
        players_order_in_trick = np.arange(
            self.trick_starting_player_idx,
            self.trick_starting_player_idx + PLAYER_COUNT
        ) % PLAYER_COUNT

        trick_ordered = []
        for player_idx in players_order_in_trick:
            base_idx = STATE_IDX_HANDS_INFO_START[player_idx]
            player_hand = self._np_state[base_idx:base_idx + 52]
            trick_raw_idx = np.where(player_hand == 2)[0]
            if len(trick_raw_idx) > 0:
                trick_ordered.append(trick_raw_idx[0])

        return np.array(trick_ordered)

    @property
    def points_collected(self) -> np.ndarray:
        """
        The number of points collected by each player in the round.
        Does not take the moon shot into account.
        """
        return self._np_state[STATE_IDX_POINTS_COLLECTED]

    @property
    def scores(self) -> np.ndarray:
        """
        The score of each player in the round.
        Takes the moon shot into account.
        """
        round_scores = self.points_collected
        if self.rules.moon_shot and self.is_moon_shot_triggered:
            return MAX_POINTS - round_scores
        return round_scores

    @property
    def current_player_idx(self) -> np.int_:
        """ID of the player that is expected to throw the next card"""
        return (len(self.current_trick_unordered) + self.trick_starting_player_idx) % PLAYER_COUNT

    @property
    def leading_suit(self) -> int | None:
        """Leading suit in the current trick, or None if the trick is empty"""
        if len(self.current_trick_unordered) == 0:
            return None
        lead_suit_idx = np.where(self._np_state[STATE_IDX_LEAD_SUITS] == 1)[0].item()
        return lead_suit_idx

    @property
    def is_current_trick_full(self) -> bool:
        return len(self.current_trick_unordered) == PLAYER_COUNT

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
            if self._np_state[STATE_IDX_HANDS_INFO_START[player_idx]] == 1:
                self._np_state[STATE_IDX_TRICK_STARTING_PLAYER] = player_idx

    def get_hand(self, player_idx: int | np.int_) -> np.ndarray:
        base_idx = STATE_IDX_HANDS_INFO_START[player_idx]
        hand_mask = self._np_state[base_idx:(base_idx + 52)]
        return np.where(hand_mask == 1)[0]

    def override_hand(self, player_idx: int | np.int_, hand: np.ndarray) -> None:
        """
        Use with caution and only at the start of the game.
        This is really exposed only for the card pass env.
        """
        base_idx = STATE_IDX_HANDS_INFO_START[player_idx]
        self._np_state[base_idx:(base_idx + 52)] = 0
        self._np_state[base_idx + hand] = 1

    def __can_perform_card_passing(self) -> bool:
        """
        Returns ``True`` if a passing cards can be performed
        """
        if self.pass_direction == PassDirection.NO_PASSING:
            return False
        if not self.rules.passing_cards:
            return False
        return not self.are_cards_passed

    def pick_cards_to_pass(self, player_idx: int | np.int_, cards_idx: np.ndarray):
        if not self.__can_perform_card_passing():
            return
        assert np.all(self._np_state[STATE_IDX_HANDS_INFO_START[player_idx] + cards_idx] == 1)
        self.__cards_to_pass[player_idx] = cards_idx

    def perform_cards_passing(self):
        if not self.__can_perform_card_passing():
            return

        if any([len(cards) != CARDS_TO_PASS_COUNT for cards in self.__cards_to_pass]):
            raise ValueError('Cannot pass the cards if not all players selected '
                             f'their {CARDS_TO_PASS_COUNT} cards to pass.')

        pass_offset = [1, 3, 2][self.pass_direction.value]

        for player_idx in range(PLAYER_COUNT):
            target_idx = (player_idx + pass_offset) % PLAYER_COUNT
            cards_to_pass_idx = self.__cards_to_pass[player_idx]
            self._np_state[STATE_IDX_HANDS_INFO_START[player_idx] + cards_to_pass_idx] = 0
            self._np_state[STATE_IDX_HANDS_INFO_START[target_idx] + cards_to_pass_idx] = 1

        self.are_cards_passed = True
        # just in case 2 of clubs was passed
        self.set_starting_player()

    def play_card(self, card_idx: int | np.int_):
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

        if self.leading_suit is None:
            self._np_state[STATE_IDX_LEAD_SUITS[card_idx // 13]] = 1

        self._np_state[STATE_IDX_HANDS_INFO_START[self.current_player_idx] + card_idx] = 2

        if is_heart(card_idx) and not self.are_hearts_broken:
            self._np_state[STATE_IDX_ARE_HEARTS_BROKEN] = 1

    def complete_trick(self, return_ordered=False) -> tuple[np.ndarray, np.int32]:
        """
        Complete the current trick and prepare for the next one

        Args:
            return_ordered: If True, the trick will be returned in the order
                that the cards were played. If False, the trick will be
                ordered according to the order of the player list. The
                former is more natural and user-friendly, while the latter
                is computationally cheaper and makes it easier to track
                cards played by each particular player

        Returns:
            A tuple of two elements, the first one is the trick content, and
            the second is the index of a player who took the trick.
        """
        if not self.is_current_trick_full:
            raise RuntimeError('The trick is not full and therefore cannot be completed')

        if return_ordered:
            trick_to_return = self.current_trick_ordered
        else:
            trick_to_return = self.current_trick_unordered

        trick = self.current_trick_unordered
        winner_idx = get_winning_card_argmax(
            cards=trick,
            leading_suit=self.leading_suit,
        )
        pts = sum(points_for_card(c) for c in trick)

        self._np_state[STATE_IDX_TRICK_NO] += 1
        hands = self._np_state[1:209]
        hands[hands == 2] = -1
        self._np_state[STATE_IDX_LEAD_SUITS] = 0
        self._np_state[STATE_IDX_POINTS_COLLECTED[winner_idx]] += pts
        self._np_state[STATE_IDX_TRICK_STARTING_PLAYER] = winner_idx
        self._np_state[STATE_IDX_TAKEN_CARDS[trick]] = winner_idx

        return trick_to_return, winner_idx

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
            random_state=self._next_seed(),
        )
        return next_round

    def __deepcopy__(self, memo):
        """
        Note:
            only things that need to be deep-copied for the environments are copied.
        """
        result = HeartsRound.__new__(HeartsRound)
        result._np_state = self._np_state.copy()
        result.are_cards_passed = True
        result.rules = self.rules
        return result
