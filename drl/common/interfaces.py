# drl/common/interfaces.py
import torch
import torch.nn as nn
from torch import Tensor
from abc import ABC, abstractmethod
from drl.common.types import Observation, Action, Reward, Done, PolicyLogits, QValues, Value


# ---------------------------------------------------------
# Environment interface
# ---------------------------------------------------------
class Environment(ABC):
    """
    Base class for vectorized RL environments.

    Required semantics:
    - maintains a batch of B parallel environments
    - internal state lives inside the env object
    - reset_state(idx) allows resetting selected envs
    - apply(action) updates internal state, returns rewards and termination flags
    - get_obs() returns current observations
    """

    # ---------------------------------------------------
    # required abstract properties
    # ---------------------------------------------------
    @property
    @abstractmethod
    def batch_size(self) -> int: ...

    @property
    @abstractmethod
    def random_termination(self) -> bool: ...

    @property
    @abstractmethod
    def gamma(self) -> float: ...

    @property
    @abstractmethod
    def obs_template(self) -> Observation: ...

    @property
    @abstractmethod
    def num_actions(self) -> int: ...

    # ---------------------------------------------------
    # required abstract methods
    # ---------------------------------------------------

    @abstractmethod
    def reset_state(self, mask: Tensor) -> None:
        """
        Reset the environments selected by mask.
        mask: BoolTensor, shape (B,)
        """

    @abstractmethod
    def apply(self, action: Action) -> tuple[Reward, Done]:
        """
        Apply batched actions and update internal state.

        Args:
            action: LongTensor shape (B,)
        Returns:
            reward: FloatTensor shape (B,)
            done: FloatTensor binary mask shape (B,)
        """

    @abstractmethod
    def get_obs(self) -> Observation:
        """
        Return current observation from internal state.
        shape: (B, *obs_template)
        """

    # ---------------------------------------------------
    # default control flow methods
    # ---------------------------------------------------
    def reset(self) -> Observation:
        """
        Reset all environments in the batch.
        Return current observations
        """
        mask = torch.ones(self.batch_size, dtype=torch.bool)
        self.reset_state(mask)
        return self.get_obs()

    def step(self, action: Action) -> tuple[Observation, Reward, Done]:
        # apply environment transition
        reward, env_done = self.apply(action)
        # conditionally, represent discounting with random termination.
        if self.random_termination:
            # gamma is the standard RL discount factor interpreted here as survival prob
            rand_done = (torch.rand(self.batch_size, device=reward.device) < (1 - self.gamma)).float()
            done = torch.maximum(env_done, rand_done)
        else:
            done = env_done
        # Reset every state where done is 1.0. This is consistent with env termination + random termination
        reset_mask = done > 0.5
        if reset_mask.any():
            self.reset_state(reset_mask)
        # whether we used random termination or not, this is the correct output
        return self.get_obs(), reward, done


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
    model: nn.Module

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
