# drl/common/interfaces.py

import torch.nn as nn
from abc import ABC, abstractmethod
from drl.common.types import Observation, Action, Reward, Done, PolicyLogits, QValues, Value


# ---------------------------------------------------------
# Environment interface
# ---------------------------------------------------------

class Environment(ABC):
    """
    Base class for environments.
    User implementation must follow this contract.
    """

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
# Agent interface
# ---------------------------------------------------------

class Agent(ABC):
    """
    Base class for agents.
    Responsibilities:
    - expose rollout_step(): produces action + intermediates for training
    - expose act(): produces only the action for evaluation / deployment
    - implicit: store model attribute (nn.Module)
    """

    @abstractmethod
    def rollout_step(self, obs: Observation):
        """
        Used during RL experience collection.

        Returns:
            action: (B,) tensor
            intermediates: model outputs needed for training
        """
        pass

    @abstractmethod
    def act(self, obs: Observation) -> Action:
        """
        Used during evaluation/deployment.

        Returns:
            action: (B,) tensor
        """
        pass

    def reset(self) -> None:
        """
        Reset internal state (e.g. RNN hidden states).
        Default is no-op.
        """
        pass


# ---------------------------------------------------------
# Model interfaces (forward signatures only)
# ---------------------------------------------------------

class PolicyValueModel(nn.Module, ABC):
    """forward(obs) -> (policy_logits, value)"""

    @abstractmethod
    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        ...


class QValueModel(nn.Module, ABC):
    """forward(obs) -> q_values"""

    @abstractmethod
    def forward(self, obs: Observation) -> QValues:
        ...


class PolicyOnlyModel(nn.Module, ABC):
    """forward(obs) -> policy_logits"""

    @abstractmethod
    def forward(self, obs: Observation) -> PolicyLogits:
        ...
