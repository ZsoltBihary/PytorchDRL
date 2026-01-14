import torch
import torch.nn as nn
from drl.common.types import Observation, PolicyLogits, Value
from drl.common.interfaces import PolicyValueModel


class ToyTradingModelMLP(PolicyValueModel):
    def __init__(self, obs_template: Observation, action_dim: int, hidden_dim: int = 32):
        super().__init__()
        cat_dim = obs_template.price.size(-1) + 1
        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_features=cat_dim, out_features=hidden_dim),
            nn.ReLU(),
            # nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU())
        # policy head
        self.policy_head = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=action_dim))
        # value head
        self.value_head = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=1))

    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        # stack obs info
        x = torch.cat((obs.price / 100.0, obs.pos[:, None]), dim=1)  # This did not work !!!
        # x = torch.cat((obs[0], obs[1][:, None]), dim=1)
        x = self.trunk(x)
        logits = self.policy_head(x)            # shape: (B, A)
        value = self.value_head(x).squeeze(-1)  # shape: (B,)
        return logits, value
