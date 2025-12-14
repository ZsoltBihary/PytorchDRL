# algorithms/ppo/interfaces.py

from abc import ABC, abstractmethod
from torch import Tensor
from algorithms.common.typing import Observation, Action, Reward, Done, PolicyLogits, Value
LogProb = Tensor          # shape (B,)
Entropy = Tensor          # shape (B,)


class ActorCriticNet(ABC):
    """Abstract interface for PPO actor-critic networks.

    Expected for discrete action spaces.
    Concrete networks should use torch.distributions.Categorical
    to sample actions and compute log probabilities.
    """

    @abstractmethod
    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        """
        Forward pass.

        Args:
            obs: Observation tensor of shape (B, ...)

        Returns:
            policy_logits: Tensor of shape (B, num_actions)
            value: Tensor of shape (B,)
        """
        ...

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """
        Sample actions given observations.

        For discrete actions, typically:
            dist = Categorical(logits=policy_logits)
            action = dist.sample()

        Returns:
            action: Tensor of shape (B,)
        """
        ...

    @abstractmethod
    def evaluate_action(self, obs: Observation, action: Action) -> tuple[LogProb, Entropy, Value]:
        """
        Evaluate actions for PPO loss.

        Args:
            obs: Observation tensor of shape (B, ...)
            action: Tensor of shape (B,)

        Returns:
            log_prob: Tensor of shape (B,) - log probability of each action
            entropy: Tensor of shape (B,) - entropy of policy for each batch
            value: Tensor of shape (B,) - critic value estimates
        """
        ...


class RolloutBuffer(ABC):
    """Abstract interface for PPO rollout buffers."""

    @abstractmethod
    def add(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        done: Done,
        value: Value,
        log_prob: LogProb
    ) -> None:
        """Add one timestep of data to the buffer."""
        ...

    @abstractmethod
    def compute_returns_and_advantages(self, gamma: float, lam: float) -> None:
        """
        Compute discounted returns and Generalized Advantage Estimation (GAE)
        for the entire buffer.
        """
        ...

    @abstractmethod
    def get_batches(self, batch_size: int):
        """
        Yield mini-batches for PPO update.

        Each batch should contain:
            obs, actions, returns, advantages, old_log_probs
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear the buffer after an update."""
        ...
