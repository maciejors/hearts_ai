import numpy as np

from hearts_ai.constants import Suit
from hearts_ai.game.deck import Card
from hearts_ai.players.base.base_player import BasePlayer
from hearts_ai.utils import points_for_card


class InputPlayer(BasePlayer):
    """A player with console input"""

    @staticmethod
    def _sorted_hand(hand: list[Card]) -> list[Card]:
        suit_order = {
            Suit.CLUB: 0,
            Suit.DIAMOND: 1,
            Suit.SPADE: 2,
            Suit.HEART: 3,
        }
        return sorted(hand, key=lambda card: (suit_order[card.suit], card.rank))

    @staticmethod
    def pretty_print_hand(hand: list[Card], valid_cards_idx: list[int] | None = None) -> None:
        if valid_cards_idx is None:
            valid_cards_idx = list(range(len(hand)))

        numbers_row = ' | '.join(
            f'{valid_cards_idx.index(i) + 1:^3}'
            if i in valid_cards_idx else ' ' * 3
            for i in range(len(hand))
        )

        cards_row = ' | '.join(f'{str(card):^3}' for card in hand)

        print(numbers_row)
        print(cards_row)

    def play_card(self,
                  hand: list[Card],
                  trick: list[Card | None],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        print()

        if len(trick) == 0:
            print('You are leading the trick')
        else:
            print(f'Current trick: {', '.join([str(card) for card in trick])}')

        print('Your hand:')
        hand = self._sorted_hand(hand)
        valid_cards = self._get_valid_plays(hand, trick, are_hearts_broken, is_first_trick)
        valid_cards_idx = [hand.index(card) for card in valid_cards]
        self.pretty_print_hand(hand, valid_cards_idx)

        while True:
            try:
                choice = int(input(f'Choose a card (1-{len(valid_cards)}): ')) - 1
                if 0 <= choice < len(valid_cards):
                    return valid_cards[choice]
                else:
                    print('Choice out of range.')
            except ValueError:
                print('Please enter a number.')

    def select_cards_to_pass(self, hand: list[Card]) -> list[Card]:
        hand = self._sorted_hand(hand)
        print('Passing cards. Your hand:')
        self.pretty_print_hand(hand)

        while True:
            try:
                input_raw = input(f'Select 3 cards (1-{len(hand)}), separated by commas (e.g. 1,3,6): ')
                selected_indices = [int(choice.strip()) - 1 for choice in input_raw.split(',')]

                if len(selected_indices) != 3:
                    print('You need to pass exactly 3 cards.')
                    continue
                if len(selected_indices) != len(np.unique(selected_indices)):
                    print('Selections must be unique. Please try again')
                    continue

                return [hand[i] for i in selected_indices]

            except ValueError:
                print('Invalid input. Please try again.')
            except IndexError:
                print('Choice(s) out of bounds. Please try again.')

    def post_trick_callback(self, trick: list[Card | None], is_trick_taken: bool) -> None:
        print(f'Trick outcome: {', '.join([str(card) for card in trick])} '
              f'({sum([points_for_card(c) for c in trick])} pts)')
        if is_trick_taken:
            print('You take this trick.')
        input('Press [Enter] to proceed ')

    def post_round_callback(self, score: int) -> None:
        print(f'Round finished. Your score: {score}')
