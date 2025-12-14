# algorithms/common/typing.py
# Semantic tensor aliases shared across DRL algorithms

from torch import Tensor

Observation = Tensor     # shape (B, ...)
Action = Tensor          # shape (B,)
Reward = Tensor          # shape (B,)
Done = Tensor            # shape (B,)
PolicyLogits = Tensor    # shape (B, num_actions)
Value = Tensor           # shape (B,)
