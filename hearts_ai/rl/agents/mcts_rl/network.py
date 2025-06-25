import torch
import torch.nn as nn


class MCTSRLNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super(MCTSRLNetwork, self).__init__()
        hidden_layer_size = 128
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_layer_size, n_actions)
        self.value_head = nn.Linear(hidden_layer_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(obs)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value.squeeze(-1)
