# worlds/multitrading/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Depthwise convolution along (S, T), shared across K
# Each factor K shares the same convolution weights
# ============================================================
class DepthwiseST(nn.Module):
    def __init__(self, C, kernel_size=(3, 3)):
        """
        C: number of feature channels
        kernel_size: (S, T) kernel size
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=C,   # depthwise per feature
            bias=False
        )

    def forward(self, x):
        """
        x: (B, K, C, S, T)
        returns: (B, K, C, S, T)
        """
        B, K, C, S, T = x.shape

        # Merge B and K into batch
        x = x.reshape(B * K, C, S, T)   # (B*K, C, S, T)

        # Apply depthwise conv2d (shared across K)
        x = self.conv(x)                # (B*K, C, S, T)

        # Restore original shape
        return x.reshape(B, K, C, S, T)


# ============================================================
# Pointwise mixing along N <-> K (assets <-> factors)
# ============================================================
class PointwiseN(nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=N_in,
            out_channels=N_out,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        """
        x:       (B, N_in, C, S, T)
        returns: (B, N_out, C, S, T)
        """
        return self.conv(x)


# ============================================================
# Pointwise mixing along C (feature channels)
# ============================================================
class PointwiseC(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        """
        x: (B, K, C_in, S, T)
        returns: (B, K, C_out, S, T)
        """
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, K, S, T)
        x = self.conv(x)
        return x.permute(0, 2, 1, 3, 4)


# ============================================================
# Channel-wise LayerNorm (batch-independent)
# ============================================================
class ChannelLayerNorm(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ln = nn.LayerNorm(C)

    def forward(self, x):
        """
        x: (B, K, C, S, T)
        """
        x = x.permute(0, 1, 3, 4, 2)  # (B, K, S, T, C)
        x = self.ln(x)
        return x.permute(0, 1, 4, 2, 3)


# ============================================================
# Shared additive modulation from position (inventory)
# ============================================================
class PositionBias(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(1, C, bias=False)

    def forward(self, position):
        """
        position: (B, N)
        returns:  (B, N, C, 1, 1)
        """
        pos = position.unsqueeze(-1)          # (B, N, 1)
        bias = self.linear(pos)                # (B, N, C)
        return bias.unsqueeze(-1).unsqueeze(-1)


# ============================================================
# Residual Block
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, N, K, C, CC, N2K_shared, kernel_size=(3, 3)):
        super().__init__()

        # Factor handling
        self.N2K = N2K_shared          # shared across blocks
        self.K2N = PointwiseN(K, N)    # block-specific

        # Residual path
        self.dw1 = DepthwiseST(C, kernel_size)
        self.pw1 = PointwiseC(C, CC)
        self.ln1 = ChannelLayerNorm(CC)
        self.relu = nn.ReLU()

        self.dw2 = DepthwiseST(CC, kernel_size)
        self.pw2 = PointwiseC(CC, C)
        self.ln2 = ChannelLayerNorm(C)

    def forward(self, x, position_bias):
        """
        x:             (B, N, C, S, T)
        position_bias: (B, N, C, 1, 1)
        """
        # N -> K
        h = self.N2K(x)

        # Residual branch
        h = self.dw1(h)
        h = self.pw1(h)
        h = self.ln1(h)
        h = self.relu(h)

        h = self.dw2(h)
        h = self.pw2(h)

        # K -> N
        h = self.K2N(h)

        # Residual + norm + position modulation
        h = self.ln2(x + h)
        h = h + position_bias
        return self.relu(h)


# ============================================================
# Pooling block (along S and/or T)
# ============================================================
class AvgPoolST(nn.Module):
    """
    Average pooling over S and/or T using AvgPool2d.
    """

    def __init__(self, pool_s=1, pool_t=1):
        super().__init__()
        self.pool_s = pool_s
        self.pool_t = pool_t
        self.pool = nn.AvgPool2d(
            kernel_size=(pool_s, pool_t),
            stride=(pool_s, pool_t)
        )

    def forward(self, x):
        """
        x: (B, N, C, S, T)
        """
        B, N, C, S, T = x.shape
        x = x.reshape(B * N, C, S, T)
        x = self.pool(x)
        S_new, T_new = x.shape[-2:]
        return x.reshape(B, N, C, S_new, T_new)


# ============================================================
# Full Trunk
# ============================================================
class TradingTrunk(nn.Module):
    """
    block_list example:
    ['R', 'T', 'R', 'T', 'R', 'S', 'R']
    """

    def __init__(self, N, K, C, CC, block_list):
        super().__init__()

        self.shared_N2K = PointwiseN(N, K)
        self.position_bias = PositionBias(C)

        self.layers = nn.ModuleList()

        for token in block_list:
            if token == 'R':
                self.layers.append(
                    ResidualBlock(
                        N=N, K=K, C=C, CC=CC,
                        N2K_shared=self.shared_N2K
                    )
                )
            elif token == 'T':
                self.layers.append(AvgPoolST(pool_s=1, pool_t=2))
            elif token == 'S':
                self.layers.append(AvgPoolST(pool_s=2, pool_t=1))
            else:
                raise ValueError(f"Unknown block token: {token}")

    def forward(self, x, position):
        """
        x:        (B, N, C, S, T)
        position: (B, N)
        """
        pos_bias = self.position_bias(position)

        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, pos_bias)
            else:
                x = layer(x)

        return x


# ============================================================
# Importance-weighted aggregation
# ============================================================
class ImportanceAggregator(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.importance = nn.Conv2d(C, 1, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: (B, N, C, S, T)
        returns: (B, N, C)
        """
        B, N, C, S, T = x.shape
        x_ = x.reshape(B * N, C, S, T)

        logits = self.importance(x_)
        weights = F.softmax(logits.flatten(-2), dim=-1)
        weights = weights.view(B * N, 1, S, T)

        pooled = (x_ * weights).sum(dim=(-2, -1))
        return pooled.view(B, N, C)


# ============================================================
# Policy Head
# ============================================================
class PolicyHead(nn.Module):
    """
    Outputs action logits:
    (B, N, A)
    """

    def __init__(self, C, A):
        super().__init__()
        self.A = A

        self.agg = ImportanceAggregator(C)
        self.fc_feat = nn.Linear(C, C, bias=False)
        self.fc_pos = nn.Linear(1, C, bias=False)
        self.fc_out = nn.Linear(C, 1, bias=False)

    def forward(self, x, position, delta_positions):
        """
        x:               (B, N, C, S, T)
        position:        (B, N)
        delta_positions: (A,)
        """
        feat = self.fc_feat(self.agg(x))      # (B, N, C)

        new_pos = position.unsqueeze(-1) + delta_positions.view(1, 1, self.A)
        pos_emb = self.fc_pos(new_pos.unsqueeze(-1))  # (B, N, A, C)

        feat = feat.unsqueeze(2)              # (B, N, 1, C)
        h = F.relu(feat + pos_emb)

        return self.fc_out(h).squeeze(-1)      # (B, N, A)


# ============================================================
# Value Head
# ============================================================
class ValueHead(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.agg = ImportanceAggregator(C)
        self.fc_feat = nn.Linear(C, C, bias=False)
        self.fc_pos = nn.Linear(1, C, bias=False)
        self.fc_out = nn.Linear(C, 1, bias=False)

    def forward(self, x, position):
        """
        x:        (B, N, C, S, T)
        position: (B, N)
        """
        feat = self.fc_feat(self.agg(x))
        pos_emb = self.fc_pos(position.unsqueeze(-1))
        h = F.relu(feat + pos_emb)
        h = h.mean(dim=1)
        return self.fc_out(h)


# ============================================================
# Sanity check
# ============================================================
if __name__ == "__main__":
    from torchinfo import summary

    B = 32
    N = 1      # number of assets
    K = 1
    C = 30
    CC = 15
    S = 16
    T = 64
    A = 5       # number of actions

    block_list = ['R', 'T', 'R', 'T', 'R', 'S', 'R']
    deltas = torch.arange(-(A // 2), A // 2 + 1) * 0.5 / N

    trunk = TradingTrunk(N=N, K=K, C=C, CC=CC, block_list=block_list)
    pol_head = PolicyHead(C=C, A=A)
    val_head = ValueHead(C=C)

    x = torch.randn(B, N, C, S, T)
    pos = torch.randn(B, N)

    out = trunk(x, pos)
    logits = pol_head(out, pos, deltas)
    val = val_head(out, pos)

    # Print model summaries
    print("\n*************** TRUNK ***************")
    summary(trunk, input_data=(x, pos),
            col_names=["input_size", "output_size", "num_params"], depth=3, verbose=1)

    print("\n*************** POLICY HEAD ***************")
    summary(pol_head, input_data=(out, pos, deltas),
            col_names=["input_size", "output_size", "num_params"], depth=3, verbose=1)

    print("\n*************** VALUE HEAD ***************")
    summary(val_head, input_data=(out, pos),
            col_names=["input_size", "output_size", "num_params"], depth=3, verbose=1)

    print("B, N, K, C,CC, S, T, A =")
    print(B, N, K, C, CC, S, T, A)

    print("Shapes:")
    print("x:", x.shape)
    print("out:", out.shape)
    print("logits:", logits.shape)
    print("value:", val.shape)
