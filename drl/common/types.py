# drl/common/types.py
# Semantic tensor aliases shared across drl/

from torch import Tensor

# Observation = Tensor

# Recursive tensor tree type
TensorTree = Tensor | tuple["TensorTree", ...]

Observation = TensorTree   # shape (B, ...) for all components
Action = Tensor            # shape (B,)
Reward = Tensor            # shape (B,)
Done = Tensor              # shape (B,)
PolicyLogits = Tensor      # shape (B, num_actions)
QValues = Tensor           # shape (B, num_actions)
Value = Tensor             # shape (B,)
LogProb = Tensor           # shape (B,)
Entropy = Tensor           # shape (B,)
