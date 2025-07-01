from collections.abc import Iterable
from typing import TypeVar, Callable

import numpy as np

_ActTypeGeneric = TypeVar('_ActTypeGeneric')
_ObsTypeGeneric = TypeVar('_ObsTypeGeneric')
ActionTakingCallback = Callable[[_ObsTypeGeneric, np.ndarray], _ActTypeGeneric]
ActionTakingCallbackParam = ActionTakingCallback | list[ActionTakingCallback]


def handle_action_taking_callback_param(
        p: ActionTakingCallbackParam,
        list_len: int,
) -> list[ActionTakingCallback]:
    if isinstance(p, Iterable):
        return p
    return [p] * list_len
