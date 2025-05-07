from typing import TypeVar, Callable

import numpy as np
from gymnasium.core import ObsType, ActType

_ActTypeGeneric = TypeVar('_ActTypeGeneric')
_ObsTypeGeneric = TypeVar('_ObsTypeGeneric')
ActionTakingCallback = Callable[[_ObsTypeGeneric, list[bool]], _ActTypeGeneric]


def get_random_action_taking_callback(random_state: int) -> ActionTakingCallback:
    rng = np.random.default_rng(random_state)

    def callback(_: ObsType, action_masks: list[bool]) -> ActType:
        legal_actions = np.flatnonzero(np.array(action_masks))
        return rng.choice(legal_actions)

    return callback
