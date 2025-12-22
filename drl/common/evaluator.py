# drl/common/evaluator.py

import torch
from drl.common.interfaces import Environment, Agent


class Evaluator:
    """
    Evaluate an agent on a batch of environments, computing exact discounted returns
    from the initial states. Supports both continuing and episodic environments.
    """
    def __init__(self, env: Environment, agent: Agent, max_steps: int = 1000):
        """
        Args:
            env: Environment object (already configured for evaluation)
            agent: Agent object (with act(obs) method, read-only)
            max_steps: Maximum number of environment steps per rollout
        """
        self.env = env
        self.agent = agent
        self.gamma = env.gamma  # use env gamma
        self.max_steps = min(max_steps, round(10.0 / (1.0001 - self.gamma)))
        self.agent.model.eval()

    @torch.no_grad()
    def run(self):
        """
        Run a full evaluation rollout from initial state.

        Returns:
            returns: FloatTensor shape (B,), discounted returns for each env
        """
        B = self.env.batch_size
        returns = torch.zeros(B, dtype=torch.float32)
        discount = torch.ones(B, dtype=torch.float32)

        obs = self.env.reset()                  # reset all envs to initial state
        for t in range(self.max_steps):
            action = self.agent.act(obs)
            # step environment
            obs, reward, done = self.env.step(action)
            # accumulate discounted return
            returns += discount * reward
            # update discount factor, mask terminated envs
            discount = discount * self.gamma * (1.0 - done)

        return returns
