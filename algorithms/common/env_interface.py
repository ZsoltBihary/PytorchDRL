# algorithms/common/env_interface.py

from abc import ABC, abstractmethod
from algorithms.common.typing import Observation, Action, Reward, Done


class Environment(ABC):
    """Abstract environment interface for batched RL."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of parallel environments."""
        ...

    @property
    @abstractmethod
    def obs_shape(self) -> tuple[int, ...]:
        """
        Shape of a single observation (without batch dimension).
        Returns:
            tuple of ints representing observation dimensions, e.g., (C, L)
        """
        ...

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, Done]:
        """
        Perform one environment step.

        Returns:
            observation: Observation of shape (B, ...)
            reward: Reward of shape (B,)
            done: Done of shape (B,) indicating environment termination
        """
        ...
