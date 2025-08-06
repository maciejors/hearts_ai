from typing import TypeVar, Callable, Sequence

import numpy as np

_ActTypeGeneric = TypeVar('_ActTypeGeneric')
_ObsTypeGeneric = TypeVar('_ObsTypeGeneric')
ActionTakingCallback = Callable[[_ObsTypeGeneric, np.ndarray], _ActTypeGeneric]
ActionTakingCallbackParam = ActionTakingCallback | list[ActionTakingCallback]


def ensure_sequence[T](p: T, sequence_length: int) -> Sequence[T]:
    if isinstance(p, Sequence):
        return p
    return [p] * sequence_length
