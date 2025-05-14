import numpy as np
from gymnasium.core import ObsType, ActType

from hearts_ai.rl.env.utils import ActionTakingCallback
from .common import SupportedAlgorithm


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
