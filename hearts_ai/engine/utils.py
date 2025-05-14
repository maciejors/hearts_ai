from .constants import Suit, HEART_POINTS, Q_SPADES_POINTS, PLAYER_COUNT
from .deck import Card


def is_q_spades(card: Card) -> bool:
    return card.suit == Suit.SPADE and card.rank == 'Q'


def is_heart(card: Card) -> bool:
    return card.suit == Suit.HEART


def is_starting_card(card: Card) -> bool:
    return card.suit == Suit.CLUB and card.rank == '2'


def points_for_card(card: Card) -> int:
    if is_heart(card):
        return HEART_POINTS
    if is_q_spades(card):
        return Q_SPADES_POINTS
    return 0


def get_trick_winner_idx(trick: list[Card], trick_starting_player_idx: int = 0) -> int:
    winning_card = trick[0]
    winner_idx = trick_starting_player_idx

    for i in range(1, PLAYER_COUNT):
        player_idx = (trick_starting_player_idx + i) % PLAYER_COUNT
        card = trick[i]

        if card.suit == trick[0].suit and card.rank_value > winning_card.rank_value:
            winning_card = card
            winner_idx = player_idx

    return winner_idx


def get_valid_plays(hand: list[Card],
                    trick: list[Card],
                    are_hearts_broken: bool,
                    is_first_trick: bool) -> list[Card]:
    if len(trick) == 0 and is_first_trick:
        # return a list with just the two of clubs
        return [card for card in hand
                if is_starting_card(card)]

    if len(trick) == 0:
        if not are_hearts_broken:
            non_hearts = [card for card in hand if not is_heart(card)]
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
            if not is_heart(card) and not is_q_spades(card)
        ]
        if len(non_points) > 0:
            return non_points

    return hand.copy()
