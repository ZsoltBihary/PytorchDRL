# drl/agents/policy_only_agent.py

# import torch
# from torch.distributions import Categorical
# from drl.common.interfaces import Agent, PolicyOnlyModel
# from drl.common.types import Observation, Action


# class PolicyOnlyAgent(Agent):
#     """
#     Agent using a PolicyOnlyModel to select actions.
#     Supports:
#     - stochastic acting via temperature-scaled sampling
#     - deterministic acting via argmax
#     """
#
#     def __init__(
#         self,
#         model: PolicyOnlyModel,
#         *,
#         temperature: float = 1.0,
#         deterministic: bool = False,
#     ) -> None:
#         self.model = model
#         self.temperature = temperature
#         self.deterministic = deterministic
#
#     def set_deterministic(self, deterministic: bool = True) -> None:
#         self.deterministic = deterministic
#
#     def act(self, obs: Observation) -> Action:
#         logits = self.model.forward(obs)
#
#         if self.deterministic:
#             return torch.argmax(logits, dim=-1)
#
#         dist = Categorical(logits=logits / self.temperature)
#         return dist.sample()
