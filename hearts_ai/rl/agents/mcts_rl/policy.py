import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.core import ObsType
from sb3_contrib.common.maskable.utils import get_action_masks


class MCTSNode:
    """
    Represents a node in MCTS

    Args:
        obs: environment state associated with the node
        env: A copy of the environment, which acts as a snapshot of the
            environment at the state ``obs``. This needs to be provided to
            allow for simulations.
        parent: Parent node (``None`` for the root node)
        prior_prob: Prior probability resulting from policy prediction
            in the network
    """

    def __init__(
            self,
            obs: ObsType,
            env: gym.Env,
            is_terminal: bool = False,
            parent: 'MCTSNode' = None,
            prior_prob: float = 0
    ):
        self.obs = obs
        self.env = env
        self.is_terminal = is_terminal
        self.parent = parent
        self.prior_prob = prior_prob
        self.children: dict[int, 'MCTSNode'] = {}
        self._visit_count = 0
        self._value_sum = 0

    def add_visit(self, value: float):
        self._visit_count += 1
        self._value_sum += value

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @property
    def value(self) -> float:
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def __str__(self):
        return ('MCTSNode{'
                f'visit_count={self.visit_count}, '
                f'value={self.value}, '
                f'len(children)={len(self.children)}'
                '}')

    def __repr__(self):
        return str(self)


class MCTSRLPolicy:
    """
    MCTS Policy enhanced with a neural network. This is a policy designed
    according to AlphaGo Zero's paper.

    MCTS consists of four steps:
    1. Selection - the tree is traversed using some policy, until a leaf node is reached;
    2. Expansion - a child node for a leaf is added
    3. Simulation - the episode is simulated until the end to obtain a reward
    4. Backpropagation - reward from simulation is propagated up towards the root

    In this implementation:
    1. Selection - the policy used is UCT, however when visit count is zero, the node
        will always be explored
    2. Expansion - all child nodes for a leaf are added
    3. Simulation - replaced by a call to a neural network, estimating a value of a state
    4. Backpropagation - proceeds as normal

    Args:
        network: A neural network architecture to use
        n_actions: Number of actions in the environment
        n_simulations: Number of simulations
        c: A UCT parameter controlling the exploration-exploitation tradeoff
        device: for torch
    """

    def __init__(
            self,
            network: nn.Module,
            n_actions: int,
            n_simulations: int = 50,
            c: float = 1,
            device: str = 'cpu',
            seed: int | None = None
    ):
        self.network = network
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.c = c
        self.device = device
        self._np_random = np.random.default_rng(seed)

    def __obs_to_tensor(self, obs: ObsType) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32) \
            .unsqueeze(0) \
            .to(self.device)

    def predict(
            self,
            obs: ObsType,
            env_deepcopy: gym.Env,
            action_masks: np.ndarray,
            deterministic: bool,
    ):
        """
        Pick the best action according to current policy
        """
        root = MCTSNode(obs, env_deepcopy)
        self._expansion(root)

        for _ in range(self.n_simulations):
            # 1. selection
            node = root
            path = [node]
            while len(node.children) > 0:
                node = self._selection(node)
                path.append(node)

            # 2. expansion
            if not node.is_terminal:
                self._expansion(node)

            # 3. simulation - in AlphaGo Zero this is replaced by a neural network
            value = self._network_evaluate(node.obs)

            # 4. backpropagation
            for visited_node in reversed(path):
                visited_node.add_visit(value)

        # pick an action based on visits count
        actions_list = list(range(self.n_actions))
        visit_counts = np.array(
            [root.children[act].visit_count if act in root.children else 0
             for act in actions_list]
        )
        policy_target = visit_counts / np.sum(visit_counts)

        if deterministic:
            action = np.argmax(policy_target)
        else:
            action = self._np_random.choice(actions_list, p=policy_target)

        return action, policy_target

    def _expansion(self, node: MCTSNode):
        """
        Performs the expansion step from MCTS.
        """
        obs_tensor = self.__obs_to_tensor(node.obs)
        with torch.no_grad():
            policy_logits, _ = self.network(obs_tensor)

        action_masks = get_action_masks(node.env)
        # probs for illegal actions should be 0
        action_masks_tensor = torch.tensor(action_masks, device=policy_logits.device) \
            .unsqueeze(0)
        policy_logits[~action_masks_tensor] = -np.inf

        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]

        legal_actions = np.arange(self.n_actions)[action_masks]
        for action in legal_actions:
            env = copy.deepcopy(node.env)
            obs, _, done, _, _ = env.step(action)
            node.children[action] = MCTSNode(
                obs=obs,
                env=env,
                parent=node,
                is_terminal=done,
                prior_prob=policy_probs[action].item(),
            )

    def _selection(self, node: MCTSNode) -> MCTSNode:
        """
        Performs the selection step at a given node using UCT
        """
        total_visits = sum(child.visit_count for child in node.children.values())
        best_uct = -np.inf
        best_child = None

        for child in node.children.values():
            if child.visit_count == 0:
                uct = np.inf  # otherwise division by 0
            else:
                under_sqrt = np.log(total_visits) / child.visit_count
                uct = child.value + self.c * child.prior_prob * np.sqrt(under_sqrt)
            if uct > best_uct:
                best_uct = uct
                best_child = child

        return best_child

    def _network_evaluate(self, obs: ObsType) -> float:
        """
        Evaluate the state using the network
        """
        obs_tensor = self.__obs_to_tensor(obs)
        with torch.no_grad():
            _, value = self.network(obs_tensor)
        return value.item()
