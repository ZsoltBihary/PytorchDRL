# from __future__ import annotations
import torch
import torch.nn as nn
from src.config import Config, State, QValues
from line_profiler_pycharm import profile


# ==============================================================
#   Q-model with residual blocks + dueling heads
# ==============================================================
class QModel(nn.Module):
    """
    Flatten (price_seq, pos) -> Q-values.
    Input:
        state: (price_seq, pos), shapes (B,T) and (B,)
    Output:
        q_values: (B, A)
    """

    def __init__(self, conf: Config):
        super().__init__()

        input_dim = conf.window_size + 1
        hidden_dim = conf.hidden_dim
        num_blocks = conf.num_blocks
        use_layernorm = conf.use_layernorm
        dropout = conf.dropout
        dueling = conf.dueling
        num_actions = conf.num_actions

        # Initial projection to hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Residual backbone
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, use_layernorm, dropout))
        self.residual_stack = nn.Sequential(*blocks)

        # Final projection to action Q-values
        if dueling:
            self.output_layer = DuelingHead(hidden_dim, num_actions)
        else:
            self.output_layer = nn.Linear(hidden_dim, num_actions)

    # ----------------------------------------------------------
    @profile
    def forward(self, state: State) -> QValues:
        price_seq, pos = state

        x = torch.cat([price_seq / 100.0, pos.unsqueeze(-1)], dim=-1)

        x = self.input_layer(x)
        x = self.residual_stack(x)
        q_values = self.output_layer(x)

        return q_values


# ==============================================================
#   Residual Block (MLP)
# ==============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, use_layernorm: bool = False, dropout: float = 0.0):
        super().__init__()

        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.norm1 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.act = nn.ReLU()

    @profile
    def forward(self, x):
        h = x

        h = self.norm1(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)            # << recommended location

        h = self.norm2(h)
        h = self.fc2(h)

        return self.act(x + h)


# ==============================================================
#   Dueling Heads (MLP)
# ==============================================================
class DuelingHead(nn.Module):
    """
    Generic dueling head.
    Input:  h of shape (B, hidden_dim)
    Output: Q-values of shape (B, num_actions)
    """
    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    @profile
    def forward(self, h):
        V = self.value(h)                      # (B, 1)
        A = self.advantage(h)                  # (B, A)
        A_mean = A.mean(dim=-1, keepdim=True)  # (B, 1)
        Q = V + (A - A_mean)                   # (B, A)
        return Q


# ============================ SIMPLE MODEL =================================
# class QModel(nn.Module):
#     """
#     Flatten (price_seq, pos) -> Q-values.
#     Input:
#         state: (price_seq, pos), where
#             price_seq: (B, T)
#             pos: (B,)
#     Output:
#         q_values: (B, A)
#     """
#     # def __init__(self, window_size: int, n_actions: int, hidden_dim: int = 32):
#     def __init__(self, conf: Config):
#         super().__init__()
#         # ===== Use configuration parameters =====
#         self.trunk = nn.Sequential(
#             nn.Linear(conf.window_size + 1, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(conf.hidden_dim, conf.num_actions)
#         )
#
#     def forward(self, state: State) -> QValues:
#         price_seq, pos = state
#         x = torch.cat([price_seq, pos.unsqueeze(-1)], dim=-1)
#         q_values = self.trunk(x)
#         return q_values


if __name__ == "__main__":
    from src.environment import Environment
    from src.config import Config
    from torchinfo import summary

    # --- Create config and environment ---
    conf = Config(
        batch_size=512,
        window_size=1,
        rollout_steps=50,
        buffer_mult=1,
        learning_rate=0.005,

        use_ddqn=True,
        dueling=True,
        use_layernorm=False,
        dropout=0.0,
        num_blocks=3,
        hidden_dim=32
    )

    env = Environment(conf)
    state = env.get_state()  # (price_seq, pos)

    # --- Create the model ---
    model = QModel(conf)
    output = model(state)
    print(f"Output shape: {output.shape}")

    # --- Wrap model in a tiny nn.Module for torchinfo ---
    class QModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, price_seq, pos):
            return self.model((price_seq, pos))

    wrapper = QModelWrapper(model)

    # --- Run summary ---
    summary(
        wrapper,
        input_data=(state[0], state[1]),
        col_names=["input_size", "output_size", "num_params"],
        depth=4,
        verbose=1
    )
