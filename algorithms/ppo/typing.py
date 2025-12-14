# algorithms/ppo/typing.py
# Semantic tensor aliases for PPO algorithm

from torch import Tensor
LogProb = Tensor          # shape (B,)
Entropy = Tensor          # shape (B,)
