# drl/agents/policy_value_agent.py

import torch
from torch.distributions import Categorical
from drl.common.interfaces import Agent, PolicyValueModel
from drl.common.types import Observation, Action, PolicyLogits, Value


class PolicyValueAgent(Agent):
    """
    Agent using a PolicyValueModel to select actions.
    """

    def __init__(
        self,
        model: PolicyValueModel,
        *,
        temperature: float = 1.0,
        deterministic: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.deterministic = deterministic

    @torch.no_grad()
    def rollout_step(self, obs: Observation) -> tuple[Action, PolicyLogits, Value]:
        """
        Used during RL experience collection.
        Returns:
            action: (B,) tensor
            logits, value: model outputs needed for training
        """
        logits, value = self.model(obs)
        # for consistency, actor-critic rollouts use the true distribution, temperature is not needed here
        dist = Categorical(logits=logits)
        action = dist.sample()
        # intermediates used in actor-critic methods (e.g. PPO)
        return action, logits, value

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        """
        Used during evaluation/deployment.
        Returns:
            action: (B,) tensor
        """
        logits, _ = self.model(obs)
        # evaluation / deployment may sharpen action selection absolutely
        if self.deterministic:
            return torch.argmax(logits, dim=-1)
        # evaluation / deployment may sharpen action selection moderately
        dist = Categorical(logits=logits / self.temperature)
        return dist.sample()
