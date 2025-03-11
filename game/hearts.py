from enum import Enum

from players import BasePlayer
from .deck import Deck, Card


class HeartsGame:
    """
    Args:
        players: List of exactly 4 player 'brains'
        rule_moon_shot: Determines whether shooting the moon is enabled
        rule_passing_cards: Determines whether passing cards is enabled
        rule_jack_of_diamonds: Determinines whether the jack of diamonds rule is enabled
        random_state: Random seed for reproducibility
    """

    PLAYER_COUNT = 4
    HEART_POINTS = 1
    Q_SPADES_POINTS = 13
    MAX_POINTS = 26

    class PassDirection(Enum):
        LEFT = 0
        RIGHT = 1
        ACROSS = 2
        NO_PASSING = 3

    def __init__(self,
                 players: list[BasePlayer],
                 rule_moon_shot=True,
                 rule_passing_cards=True,
                 rule_jack_of_diamonds=False,
                 random_state: int | None = None):

        if len(players) != HeartsGame.PLAYER_COUNT:
            raise ValueError(f'There should be exactly {HeartsGame.PLAYER_COUNT} players')

        self.players = players
        self.deck = Deck(random_state=random_state)

        self.hearts_broken = False
        self.pass_direction_int = HeartsGame.PassDirection.LEFT.value

        self.rule_moon_shot = rule_moon_shot
        self.rule_passsing_cards = rule_passing_cards
        self.rule_jack_of_diamonds = rule_jack_of_diamonds

        # order of items in the list corresponds to the order of players
        self.hands = [[] for _ in players]
        self.collected_tricks = [[] for _ in players]
        self.current_trick = []

        # total points excluding points from the current round
        self.scoreboard = [0 for _ in players]

    @property
    def current_round_points(self) -> list[int]:
        round_scores = [0 for _ in self.players]

        for i, _ in enumerate(self.players):
            for card in self.collected_tricks[i]:
                if card.suit == Card.Suit.HEART:
                    round_scores[i] += 1
                elif card.suit == Card.Suit.SPADE and card.rank == 12:
                    round_scores[i] += 13

        if self.rule_moon_shot and HeartsGame.MAX_POINTS in round_scores:
            shooter_idx = round_scores.index(HeartsGame.MAX_POINTS)
            for i, _ in enumerate(self.players):
                if i != shooter_idx:
                    self.scoreboard[i] += HeartsGame.MAX_POINTS
                else:
                    self.scoreboard[i] -= HeartsGame.HEART_POINTS

        return round_scores

    def deal_cards(self):
        self.deck.shuffle()
        for i, _ in enumerate(self.players):
            self.hands[i] = list(self.deck.deal(13))
            self.collected_tricks[i] = []

    def pass_cards(self):
        if self.pass_direction_int == HeartsGame.PassDirection.NO_PASSING.value:
            return

        pass_offsets = [1, 3, 2]
        pass_offset = pass_offsets[self.pass_direction_int]

        # 1. select the cards to pass
        cards_to_pass = []
        for i, player in enumerate(self.players):
            selected_cards = player.select_cards_to_pass(self.hands[i])

            for card in selected_cards:
                self.hands[i].remove(card)

            cards_to_pass.append(selected_cards)

        # 2. pass the cards
        for i, _ in enumerate(self.players):
            target_idx = (i + pass_offset) % 4
            self.hands[target_idx].extend(cards_to_pass[i])

    def find_starting_player(self) -> int:
        for i, _ in enumerate(self.players):
            if any(card.suit == Card.Suit.CLUB and card.rank == 2 for card in self.hands[i]):
                return i

    def play_trick(self, starting_player: int, is_first_trick: bool) -> int:
        self.current_trick = []
        current_player = starting_player
        player_count = len(self.players)

        for _ in range(player_count):
            player = self.players[current_player]

            card = player.play_card(
                self.hands[current_player],
                self.current_trick.copy(),
                self.hearts_broken,
                is_first_trick
            )
            self.hands[current_player].remove(card)
            self.current_trick.append(card)

            if card.suit == Card.Suit.HEART and not self.hearts_broken:
                self.hearts_broken = True

            current_player = (current_player + 1) % 4

        # check who is taking the trick
        lead_suit = self.current_trick[0].suit
        winning_card = self.current_trick[0]
        winner_idx = starting_player

        for i in range(1, player_count):
            player_idx = (starting_player + i) % 4
            card = self.current_trick[i]

            if card.suit == lead_suit and card.rank > winning_card.rank:
                winning_card = card
                winner_idx = player_idx

        self.collected_tricks[winner_idx].extend(self.current_trick)
        return winner_idx

    def play_round(self):
        self.hearts_broken = False

        self.deal_cards()
        self.pass_cards()

        current_player = self.find_starting_player()

        for trick_num in range(13):
            current_player = self.play_trick(current_player, is_first_trick=trick_num == 0)

        # update the scoreboard
        round_scores = self.current_round_points
        for i, _ in enumerate(self.players):
            self.scoreboard[i] += round_scores[i]

        self.pass_direction_int = (self.pass_direction_int + 1) % 4
