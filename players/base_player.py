from abc import ABC, abstractmethod

from game.deck import Card


class BasePlayer(ABC):
    """Abstract base class for all Hearts players"""

    @abstractmethod
    def play_card(self,
                  hand: list[Card],
                  trick: list[Card | None],
                  are_hearts_broken: bool,
                  is_first_trick: bool) -> Card:
        raise NotImplementedError()

    @abstractmethod
    def select_cards_to_pass(self, hand: list[Card]) -> list[Card]:
        raise NotImplementedError()

    @staticmethod
    def _get_valid_plays(hand: list[Card],
                         trick: list[Card | None],
                         are_hearts_broken: bool,
                         is_first_trick: bool) -> list[Card]:
        if len(trick) == 0 and is_first_trick:
            # return a list with just the two of clubs
            return [card for card in hand
                    if card.suit == Card.Suit.CLUB and card.rank == 2]

        if len(trick) == 0:
            if not are_hearts_broken:
                non_hearts = [card for card in hand if card.suit != Card.Suit.HEART]
                if len(non_hearts) > 0:
                    return non_hearts
            return hand.copy()

        lead_suit = trick[0].suit
        matching_suit = [card for card in hand if card.suit == lead_suit]

        if len(matching_suit) > 0:
            return matching_suit

        if is_first_trick:
            non_points = [
                card for card in hand
                if card.suit != Card.Suit.HEART and not (card.suit == Card.Suit.SPADE and card.rank == 12)
            ]
            if len(non_points) > 0:
                return non_points

        return hand.copy()
