import os
import time
from typing import TypeVar, Type

from stable_baselines3.common.base_class import BaseAlgorithm

ExtendsBaseAlgorithm = TypeVar('ExtendsBaseAlgorithm', bound=BaseAlgorithm)


def clone_agent(
        agent_cls: Type[ExtendsBaseAlgorithm],
        agent_obj: ExtendsBaseAlgorithm
) -> ExtendsBaseAlgorithm:
    temp_filename = f'temp_{int(time.time() * 1000)}.zip'
    agent_obj.save(temp_filename)
    agent_copy = agent_cls.load(temp_filename)
    os.remove(temp_filename)
    return agent_copy
