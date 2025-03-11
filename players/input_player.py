import numpy as np

from game.deck import Card
from .base_player import BasePlayer


class InputPlayer(BasePlayer):
    """A player with console input"""

    @staticmethod
    def _sorted_hand(hand: list[Card]) -> list[Card]:
        suit_order = {
            Card.Suit.CLUB: 0,
            Card.Suit.DIAMOND: 1,
            Card.Suit.SPADE: 2,
            Card.Suit.HEART: 3,
        }
        return sorted(hand, key=lambda card: (suit_order[card.suit], card.rank))

    def play_card(self,
                  hand: list[Card],
                  trick: list[Card | None],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        if len(trick) == 0:
            print('You are leading the trick')
        else:
            print(f'Current trick: {', '.join([str(card) for card in trick])}')

        hand = self._sorted_hand(hand)
        print(f'Your hand: {', '.join([str(card) for card in hand])}')

        valid_cards = self._get_valid_plays(hand, trick, are_hearts_broken, is_first_trick)

        print('Valid cards to play:')
        for i, card in enumerate(valid_cards):
            print(f'{i + 1}: {card}')

        while True:
            try:
                choice = int(input(f'Choose a card (1-{len(hand)}): ')) - 1
                if 0 <= choice < len(valid_cards):
                    return valid_cards[choice]
                else:
                    print('Choice out of range.')
            except ValueError:
                print('Please enter a number.')

    def select_cards_to_pass(self, hand: list[Card]) -> list[Card]:
        hand = self._sorted_hand(hand)
        print('Passing cards. Your hand:')

        for i, card in enumerate(hand):
            print(f'{i + 1}: {card}')

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
