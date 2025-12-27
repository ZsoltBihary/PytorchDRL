# worlds/ipd/models/conv1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from drl.common.types import Observation, PolicyLogits, Value
from drl.common.interfaces import PolicyValueModel


class ScalarGlobalNorm1D(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, L)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class Residual1D(nn.Module):
    def __init__(self, C, bottleneck_ratio=0.5):
        super().__init__()
        C_b = int(C * bottleneck_ratio)

        self.conv1 = nn.Conv1d(C, C_b, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(C_b, C, kernel_size=3, padding=1, bias=False)

        # self.norm1 = nn.LayerNorm([C_b, L], elementwise_affine=False)       # after first conv
        # self.norm1 = nn.GroupNorm(C_b, C_b)  # after first conv
        self.norm1 = ScalarGlobalNorm1D()
        self.norm2 = ScalarGlobalNorm1D()        # after skip-addition

        # Initialize weights
        nn.init.orthogonal_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight, gain=1.0)  # second conv usually gain=1

    def forward(self, x):
        residual = x

        # First conv + LN + ReLU
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        # Second conv
        out = self.conv2(out)

        # Residual addition + LN + ReLU
        out = residual + out
        out = self.norm2(out)
        out = F.relu(out)

        return out


# -----------------------------
# General Head Module
# -----------------------------
class Head1D(nn.Module):
    """
    General head: pointwise conv1d -> linear -> linear
    The last linear layer uses a small gain for PPO stability.
    """
    def __init__(self, in_channels, hidden_dim, out_dim, L, final_gain=0.01):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(hidden_dim * L, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)
        self.final_gain = final_gain

        # Apply initialization immediately
        self._init_weights()

    def forward(self, x):
        out = F.relu(self.conv(x))          # (B, hidden, L)
        out = out.flatten(start_dim=1)      # (B, hidden * L)
        out = F.relu(self.fc1(out))         # (B, hidden)
        out = self.fc2(out)                 # (B, out_dim)
        return out

    def _init_weights(self):
        # Conv1D
        nn.init.orthogonal_(self.conv.weight, gain=nn.init.calculate_gain("relu"))
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        # fc1
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain("relu"))
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        # fc2 - final layer: small gain
        nn.init.orthogonal_(self.fc2.weight, gain=self.final_gain)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


# -----------------------------
# Actor-Critic Model
# -----------------------------
class PolicyValueConv1D(PolicyValueModel):
    def __init__(self, L, obs_channels, action_dim, *,
                 trunk_channels=32, num_res_blocks=3,
                 policy_hidden=16, value_hidden=8):
        super().__init__()
        # Input conv
        self.input_conv = nn.Conv1d(obs_channels, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.norm = ScalarGlobalNorm1D()
        # Residual trunk
        self.trunk_blocks = nn.Sequential(
            *[Residual1D(C=trunk_channels) for _ in range(num_res_blocks)]
        )
        # self.trunk_blocks = nn.Sequential(
        #     *[ResidualBlock1D(trunk_channels) for _ in range(num_res_blocks)]
        # )
        # Heads
        self.policy_head = Head1D(trunk_channels, policy_hidden, action_dim, L, final_gain=0.01)
        self.value_head = Head1D(trunk_channels, value_hidden, 1, L, final_gain=0.01)

        # Initialize input_conv separately
        nn.init.orthogonal_(self.input_conv.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        """
        obs: (B, L, obs_channels)
        returns:
            logits: (B, action_dim)
            value: (B,)
        """
        x = obs.transpose(1, 2)           # (B, C, L)
        x = self.input_conv(x)
        x = self.norm(x)
        x = F.relu(x)

        x = self.trunk_blocks(x)
        logits = self.policy_head(x)            # shape: (B, A)
        value = self.value_head(x).squeeze(-1)  # shape: (B,)
        return logits, value


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    from torchinfo import summary

    BB = 7
    LL = 10
    obs_channel = 5
    action_d = 2

    model = PolicyValueConv1D(obs_channels=obs_channel,
                              trunk_channels=16,
                              num_res_blocks=3,
                              L=LL,
                              policy_hidden=12,
                              action_dim=action_d,
                              value_hidden=4)

    dummy_obs = torch.randn(BB, LL, obs_channel)
    logits, value = model(dummy_obs)
    print("logits:", logits.shape)  # (B, action_dim)
    print("value:", value.shape)    # (B, 1)

    # Print model summary
    summary(
        model,
        input_data=dummy_obs,
        col_names=["input_size", "output_size", "num_params"],
        depth=2,
        verbose=1
    )
