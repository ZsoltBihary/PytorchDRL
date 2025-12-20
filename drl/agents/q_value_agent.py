# drl/agents/q_value_agent.py

import torch
from drl.common.interfaces import Agent, QValueModel
from drl.common.types import Observation, Action, QValues


class QValueAgent(Agent):
    """
    Agent using a QValueModel to select actions.
    """

    def __init__(
        self,
        model: QValueModel,
        *,
        epsilon: float = 0.1
    ):
        self.model = model
        self.epsilon = epsilon

    @torch.no_grad()
    def rollout_step(self, obs: Observation) -> tuple[Action, QValues]:
        """
        Used during RL experience collection.
        Returns:
            action: (B,) tensor
            q_values: model output needed for training
        """
        q_values = self.model(obs)
        B, A = q_values.shape
        # epsilon-greedy for exploration
        random_action = torch.randint(0, A, (B,), device=q_values.device)
        greedy_action = torch.argmax(q_values, dim=-1)
        mask = (torch.rand(B, device=q_values.device) < self.epsilon)
        action = torch.where(mask, random_action, greedy_action)
        # intermediate used in q-value methods (e.g. DQN)
        return action, q_values

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        """
        Evaluation-time step.
        """
        q_values = self.model(obs)
        # greedy for evaluation / deployment
        action = torch.argmax(q_values, dim=-1)
        return action
