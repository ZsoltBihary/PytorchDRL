# drl/common/interfaces.py

import torch.nn as nn
from abc import ABC, abstractmethod
from drl.common.types import Observation, Action, Reward, Done, PolicyLogits, QValues, Value


# ---------------------------------------------------------
# Agent interface
# ---------------------------------------------------------

class Agent(ABC):
    """
    An Agent is an actor that can act in an environment.
    It may be trainable, but training is not part of this interface.
    """

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """
        Select an action given an observation.
        Used during:
        - evaluation
        - deployment
        NOT used during:
        - training
        """
        ...

    def reset(self) -> None:
        """
        Reset internal state (e.g. RNN hidden states).
        Default is no-op.
        """
        pass


# ---------------------------------------------------------
# Environment interface
# ---------------------------------------------------------

class Environment(ABC):
    """Batched environment interface."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        ...

    @property
    @abstractmethod
    def obs_shape(self) -> tuple[int, ...]:
        """Shape of a single observation (without batch dimension)."""
        ...

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, Reward, Done]:
        ...


# ---------------------------------------------------------
# Model interfaces (forward signatures only)
# ---------------------------------------------------------

class PolicyValueModel(nn.Module, ABC):
    """forward(obs) -> (policy_logits, value)"""

    @abstractmethod
    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        ...


class QValueModel(ABC):
    """forward(obs) -> q_values"""

    @abstractmethod
    def forward(self, obs: Observation) -> QValues:
        ...


class PolicyOnlyModel(ABC):
    """forward(obs) -> policy_logits"""

    @abstractmethod
    def forward(self, obs: Observation) -> PolicyLogits:
        ...
