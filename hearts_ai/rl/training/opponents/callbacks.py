import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.engine import PassDirection, Suit
from hearts_ai.rl.env.utils import ActionTakingCallback, action_to_card, card_to_idx
from .rule_based import play_card_rule_based, select_cards_to_pass_rule_based
from ..common import SupportedAlgorithm


def get_callback_from_agent(agent: SupportedAlgorithm) -> ActionTakingCallback:
    def callback(obs: ObsType, action_masks: np.ndarray) -> ActType:
        return agent.predict(obs, action_masks=np.array(action_masks))[0]

    return callback


def get_random_action_taking_callback(random_state: int) -> ActionTakingCallback:
    rng = np.random.default_rng(random_state)

    def callback(_: ObsType, action_masks: np.ndarray) -> ActType:
        legal_actions = np.flatnonzero(np.array(action_masks))
        return rng.choice(legal_actions)

    return callback


def rule_based_play_callback(obs: ObsType, _: np.ndarray) -> ActType:
    trick_number = int(obs[0])
    is_first_trick = trick_number == 0

    # agent's hand
    hand = [action_to_card(i) for i in range(52) if obs[1 + i] == 1]

    # current trick & remaining cards in play
    current_trick = []
    remaining_cards_opponents = []
    for offset in [53, 105, 157]:
        for i in range(52):
            if obs[offset + i] == 2:
                current_trick.append(action_to_card(i))
            if obs[offset + i] == 1:
                remaining_cards_opponents.append(action_to_card(i))

    # if a heart was played means the hearts are broken
    are_hearts_broken = False
    for card_idx in range(52):
        card = action_to_card(card_idx)
        if card.suit == Suit.HEART:
            for player_offset in [1, 53, 105, 157]:
                if obs[player_offset + card_idx] == -1:
                    are_hearts_broken = True
                    break
        if are_hearts_broken:
            break

    # void info for each opponent
    # there is unfortunately no way to know the trick history from the observation,
    # so void information needs to be obtained by peeking at opponents' hands
    opponents_voids = {suit: [False, False, False] for suit in Suit}
    for opponent_idx, offset in enumerate([53, 105, 157]):
        for suit in Suit:
            has_suit = any(
                obs[offset + i] in {1, 2} and action_to_card(i).suit == suit
                for i in range(52)
            )
            if not has_suit:
                opponents_voids[suit][opponent_idx] = True

    card_to_play = play_card_rule_based(
        hand=hand,
        current_trick=current_trick,
        are_hearts_broken=are_hearts_broken,
        is_first_trick=is_first_trick,
        opponents_voids=opponents_voids,
        remaining_cards_opponents=remaining_cards_opponents,
    )
    return card_to_idx(card_to_play)


def rule_based_card_pass_callback(obs: ObsType, _: np.ndarray) -> ActType:
    already_picked_count = np.sum(obs[:52] == -1)
    hand = [
        action_to_card(i) for i in range(52)
        if obs[i] == 1 or obs[i] == -1
    ]

    if obs[52] == 1:
        direction = PassDirection.LEFT
    elif obs[53] == 1:
        direction = PassDirection.ACROSS
    else:
        # obs[54] == 1
        direction = PassDirection.RIGHT

    # this is computed three times but for the same hand will return the same result every time
    # it is not the most optimal but is clearer and easier to implement
    selected_cards = select_cards_to_pass_rule_based(hand, direction)
    card_to_pass = selected_cards[already_picked_count]
    return card_to_idx(card_to_pass)
