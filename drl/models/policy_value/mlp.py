# drl/models/policy_value/mlp.py

import torch.nn as nn
from drl.common.types import Observation, PolicyLogits, Value
from drl.common.interfaces import PolicyValueModel


class PolicyValueMLP(PolicyValueModel):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # shared body for feature extraction
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        # policy head
        self.policy_head = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        # value head
        self.value_head = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        x = self.net(obs)
        logits = self.policy_head(x)            # shape: (B, A)
        value = self.value_head(x).squeeze(-1)  # shape: (B,)
        return logits, value
