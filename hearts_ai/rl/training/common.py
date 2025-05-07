import os
import time
from datetime import datetime
from typing import TypeVar

import numpy as np
from gymnasium.core import ObsType, ActType
from sb3_contrib import MaskablePPO

from hearts_ai.rl.env import HeartsPlayEnvironment, HeartsCardsPassEnvironment
from hearts_ai.rl.env.utils import ActionTakingCallback

SupportedAlgorithm = TypeVar(
    'SupportedAlgorithm',
    MaskablePPO,
    MaskablePPO,  # this is placeholder, and will be replaced with a new algorithm later
)
SupportedEnvironment = TypeVar(
    'SupportedEnvironment',
    HeartsPlayEnvironment,
    HeartsCardsPassEnvironment,
)


def print_start_training_info(steps_per_stage: np.ndarray):
    print(f'The training starts at {datetime.now().strftime("%H:%M")}')
    print(f'It will take {int(np.sum(steps_per_stage))} steps in total')


def clone_agent(agent: SupportedAlgorithm) -> SupportedAlgorithm:
    temp_filename = f'temp_{int(time.time() * 1000)}.zip'
    agent.save(temp_filename)
    agent_copy = agent.load(temp_filename)
    os.remove(temp_filename)
    return agent_copy


def update_self_play_clones(env: SupportedEnvironment, agent: SupportedAlgorithm) -> None:
    agent_copy = clone_agent(agent)
    opponents_callbacks = [
        lambda state, action_masks: agent_copy.predict(state, action_masks=action_masks)[0]
        for _ in range(3)
    ]
    env.opponents_callbacks = opponents_callbacks


def get_random_action_taking_callback(random_state: int) -> ActionTakingCallback:
    rng = np.random.default_rng(random_state)

    def callback(_: ObsType, action_masks: list[bool]) -> ActType:
        legal_actions = np.flatnonzero(np.array(action_masks))
        return rng.choice(legal_actions)

    return callback
