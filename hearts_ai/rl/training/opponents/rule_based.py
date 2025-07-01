import numpy as np

from hearts_ai.engine import Card, Suit, PassDirection
from hearts_ai.engine.constants import CARDS_TO_PASS_COUNT
from hearts_ai.engine.utils import get_valid_plays

_Q_SPADE = Card.of('Q', Suit.SPADE)
_K_SPADE = Card.of('K', Suit.SPADE)
_A_SPADE = Card.of('A', Suit.SPADE)
_2_CLUB = Card.of('2', Suit.CLUB)
_A_CLUB = Card.of('A', Suit.CLUB)


def _filter_to_suit(cards: list[Card], suit: Suit) -> list[Card]:
    return [c for c in cards if c.suit == suit]


def _filter_out_suit(cards: list[Card], suit: Suit) -> list[Card]:
    return [c for c in cards if c.suit != suit]


def _min_by_rank(cards: list[Card], filter_to_suit: Suit | None = None) -> Card:
    if filter_to_suit is not None:
        cards = _filter_to_suit(cards, filter_to_suit)
    return min(cards, key=lambda c: c.rank_value)


def _max_by_rank(cards: list[Card], filter_to_suit: Suit | None = None) -> Card:
    if filter_to_suit is not None:
        cards = _filter_to_suit(cards, filter_to_suit)
    return max(cards, key=lambda c: c.rank_value)


def play_card_rule_based(
        hand: list[Card],
        current_trick: list[Card],
        are_hearts_broken: bool,
        is_first_trick: bool,
        opponents_voids: dict[Suit, list[bool]],
        remaining_cards_opponents: list[Card],
) -> Card:
    """
    Pick a card to play according to hard-coded rules of play.
    Implemented using guidelines from https://mark.random-article.com/hearts/hearts_tips.pdf

    Args:
        hand: player's hand
        current_trick: cards in current trick
        are_hearts_broken: whether hearts had been played
        is_first_trick: whether it is the first trick
        opponents_voids: provides known information on voids. Each value
            is a three-element list, following elements corresponding to left,
            across, and right opponent. True means the opponent has voided
            this suit, False means it may have this suit
        remaining_cards_opponents: All cards remaining in play in opponents'
            hands

    Returns:
        A card to play
    """
    leading_suit = Suit.order(current_trick[0].suit) if len(current_trick) > 0 else None
    valid_plays_sorted_ranks: list[Card] = sorted(
        [Card(i) for i in get_valid_plays(
            hand=np.array([c.idx for c in hand]),
            leading_suit=leading_suit,
            are_hearts_broken=are_hearts_broken,
            is_first_trick=is_first_trick,
        )],
        key=lambda c: c.rank_value,
    )
    if len(valid_plays_sorted_ranks) == 1:
        return valid_plays_sorted_ranks[0]

    if len(current_trick) == 0:
        spades_in_hand = _filter_to_suit(valid_plays_sorted_ranks, Suit.SPADE)
        if (len(spades_in_hand) > 0
                and _Q_SPADE in remaining_cards_opponents
                and _K_SPADE not in hand
                and _A_SPADE not in hand):
            # flush out the queen
            min_spade = spades_in_hand[0]
            return min_spade

        # maybe safe = not guaranteed to win the trick
        cards_maybe_safe = valid_plays_sorted_ranks.copy()
        # don't open with a card that is sure to win the trick
        for suit, voids in opponents_voids.items():
            # skip if you do not have this suit
            if len(_filter_to_suit(cards_maybe_safe, suit)) == 0:
                continue
            if all(voids):
                cards_maybe_safe = _filter_out_suit(cards_maybe_safe, suit)
            else:
                remaining_in_suit = _filter_to_suit(remaining_cards_opponents, suit)
                lowest_in_hand_in_suit = _min_by_rank(cards_maybe_safe, filter_to_suit=suit)
                # if the lowest in suit will take the trick anyway
                if all(lowest_in_hand_in_suit.idx >= c.idx for c in remaining_in_suit):
                    cards_maybe_safe = _filter_out_suit(cards_maybe_safe, suit)

        # don't play spades if you have dangerous spades
        if _Q_SPADE in hand or (
                _Q_SPADE in remaining_cards_opponents
                and (_K_SPADE in hand or _A_SPADE in hand)):
            if len(cards_maybe_safe) > 0:
                cards_maybe_safe_no_spades = _filter_out_suit(cards_maybe_safe, Suit.SPADE)
                if len(cards_maybe_safe_no_spades) > 0:
                    # play whatever is the lowest
                    return cards_maybe_safe_no_spades[0]
                else:
                    # no choice but to play the lowest spade
                    # if it's the queen maybe someone will snatch it
                    return cards_maybe_safe[0]
            else:
                # if we are guaranteed to win the trick and unsafe with spades, play whatever lowest but the spades
                cards_no_spades = _filter_out_suit(valid_plays_sorted_ranks, Suit.SPADE)
                if len(cards_no_spades) > 0:
                    return cards_no_spades[0]
                else:
                    # we only have spades -> play the lowest, unless it's the queen of spades,
                    # which we want to delay as much as possible
                    lowest_spade = valid_plays_sorted_ranks[0]
                    if lowest_spade != _Q_SPADE:
                        return lowest_spade
                    else:
                        return valid_plays_sorted_ranks[-1]

        # at this point it is unsafe or unnecessary to flush out the queen,
        # and we are confirmed not to have dangerous spades
        # play whatever is the lowest rank
        if len(cards_maybe_safe) > 0:
            return cards_maybe_safe[0]
        return valid_plays_sorted_ranks[0]

    # case: we do not lead the trick
    leading_suit = current_trick[0].suit
    do_we_match_suit = any(c.suit == leading_suit for c in valid_plays_sorted_ranks)
    if do_we_match_suit:
        # if it's the first trick, play the highest club
        our_highest = valid_plays_sorted_ranks[-1]
        if is_first_trick:
            return our_highest

        # if we have cards lower than the one currently winning the trick,
        # play the highest of them (unless its queen of spades)
        current_winner = _max_by_rank(current_trick, filter_to_suit=leading_suit)
        ours_lower_than_winner = [
            c for c in valid_plays_sorted_ranks
            if c.rank_value < current_winner.rank_value
        ]
        if len(ours_lower_than_winner) > 0:
            return ours_lower_than_winner[-1]

        # if we are guaranteed to win the trick, play the highest card
        remaining_in_suit = _filter_to_suit(remaining_cards_opponents, leading_suit)
        our_lowest = valid_plays_sorted_ranks[0]
        if all(our_lowest.rank_value > c.rank_value for c in remaining_in_suit):
            if our_highest != _Q_SPADE:
                return our_highest
            else:
                return valid_plays_sorted_ranks[-2]

        # otherwise, play the lowest card
        return our_lowest

    else:
        # we do not match suit
        if _Q_SPADE in valid_plays_sorted_ranks:
            return _Q_SPADE
        if _Q_SPADE in remaining_cards_opponents:
            if _A_SPADE in valid_plays_sorted_ranks:
                return _A_SPADE
            elif _K_SPADE in valid_plays_sorted_ranks:
                return _K_SPADE

        # otherwise offload a suit that we are the closest to voiding
        suits_in_hand = {s: _filter_to_suit(valid_plays_sorted_ranks, s) for s in Suit}
        suits_in_hand = {s: cards for s, cards in suits_in_hand.items() if len(cards) > 0}
        suit_closest_to_void = min(
            suits_in_hand.keys(),
            key=lambda s: len(suits_in_hand[s]),
        )
        return suits_in_hand[suit_closest_to_void][-1]


def select_cards_to_pass_rule_based(hand: list[Card], direction: PassDirection) -> list[Card]:
    """
    Pick 3 cards to pass according to hard-coded rules of play.
    Implemented using guidelines from https://mark.random-article.com/hearts/hearts_tips.pdf
    """
    picked_to_pass: set[Card] = set()
    banned_to_pass = {Card.of('A', Suit.HEART)}

    # check for emergency situation - only spade is Q of spades
    spades_on_hand = _filter_to_suit(hand, Suit.SPADE)
    if len(spades_on_hand) == 1 and _Q_SPADE in spades_on_hand:
        picked_to_pass.add(_Q_SPADE)

    if direction == PassDirection.LEFT:
        banned_to_pass.update({
            _2_CLUB, _Q_SPADE
        })
        if _A_CLUB in hand:
            picked_to_pass.add(_A_CLUB)
    else:
        if _2_CLUB in hand:
            picked_to_pass.add(_2_CLUB)
        if direction == PassDirection.RIGHT and _Q_SPADE in hand:
            picked_to_pass.add(_Q_SPADE)

    banned_to_pass.update(spades_on_hand)
    while len(picked_to_pass) < CARDS_TO_PASS_COUNT:
        remaining_cards = [c for c in hand if c not in picked_to_pass and c not in banned_to_pass]

        # try to void clubs
        if Suit.CLUB in {c.suit for c in remaining_cards}:
            clubs = _filter_to_suit(remaining_cards, Suit.CLUB)
            if len(clubs) <= CARDS_TO_PASS_COUNT - len(picked_to_pass):
                picked_to_pass.add(_max_by_rank(clubs))
                continue

        # try to void diamonds
        if Suit.DIAMOND in {c.suit for c in remaining_cards}:
            diamonds = _filter_to_suit(remaining_cards, Suit.DIAMOND)
            if len(diamonds) <= CARDS_TO_PASS_COUNT - len(picked_to_pass):
                picked_to_pass.add(_max_by_rank(diamonds))
                continue

        # get rid of high clubs
        high_clubs = [c for c in remaining_cards if c.suit == Suit.CLUB and c.rank_value >= 10]
        if high_clubs:
            picked_to_pass.add(_max_by_rank(high_clubs))
            continue

        # get rid of high cards from suit closest to voiding.
        # if equally close the priority is: clubs, diamonds, hearts
        suits_priority = [Suit.CLUB, Suit.DIAMOND, Suit.HEART]
        suits_to_consider = {
            s: _filter_to_suit(remaining_cards, s)
            for s in suits_priority if len(_filter_to_suit(remaining_cards, s)) > 0
        }
        if len(suits_to_consider) > 0:
            suit_closest_to_void = min(
                suits_to_consider.keys(),
                key=lambda suit: len(suits_to_consider[suit])
            )
            picked_to_pass.add(_max_by_rank(suits_to_consider[suit_closest_to_void]))
            continue

        # if none of the above rules work, add the highest remaining card not banned
        if remaining_cards:
            picked_to_pass.add(_max_by_rank(remaining_cards))
        else:
            # in case there are too few non-banned cards to pick from
            # could technically happen if someone has nearly only spades
            # in that case just pass the worst of them
            picked_to_pass.add(_max_by_rank(list(banned_to_pass)))

    return list(picked_to_pass)
