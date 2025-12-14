# algorithms/ppo/networks/base.py

from abc import ABC, abstractmethod
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.common.typing import Observation, Action, PolicyLogits, Value
from algorithms.ppo.typing import LogProb, Entropy


class ActorCritic(nn.Module, ABC):
    """
    Base class for actor-critic networks with discrete action spaces.

    Subclasses must implement forward(), producing policy logits and value.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        """
        Compute policy logits and value estimates.

        Args:
            obs: Observation tensor of shape (B, ...)

        Returns:
            policy_logits: Tensor of shape (B, num_actions)
            value: Tensor of shape (B,)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Policy helpers (shared across architectures)
    # ------------------------------------------------------------------
    def _distribution(self, policy_logits: PolicyLogits) -> Categorical:
        """Create a categorical policy distribution."""
        return Categorical(logits=policy_logits)

    def act(self, obs: Observation) -> tuple[Action, LogProb, Value]:
        """
        Sample actions from the policy.

        Returns:
            action: Tensor of shape (B,)
            log_prob: Tensor of shape (B,)
            value: Tensor of shape (B,)
        """
        policy_logits, value = self.forward(obs)
        dist = self._distribution(policy_logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_action(self, obs: Observation, action: Action) -> tuple[LogProb, Entropy, Value]:
        """
        Evaluate given actions under the current policy.

        Used during PPO updates.

        Returns:
            log_prob: Tensor of shape (B,)
            entropy: Tensor of shape (B,)
            value: Tensor of shape (B,)
        """
        policy_logits, value = self.forward(obs)
        dist = self._distribution(policy_logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value
