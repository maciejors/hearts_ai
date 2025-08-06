from typing import TypeVar, Callable

import numpy as np

_ActTypeGeneric = TypeVar('_ActTypeGeneric')
_ObsTypeGeneric = TypeVar('_ObsTypeGeneric')
ActionTakingCallback = Callable[[_ObsTypeGeneric, np.ndarray], _ActTypeGeneric]
ActionTakingCallbackParam = ActionTakingCallback | list[ActionTakingCallback]


def ensure_list[T](p: T, list_len: int) -> list[T]:
    if isinstance(p, list):
        return p
    return [p] * list_len
