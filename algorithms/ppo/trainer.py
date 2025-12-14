# algorithms/ppo/trainer.py

# from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from algorithms.ppo.buffer import PPORolloutBuffer
from algorithms.ppo.networks.base import ActorCritic
from algorithms.common.env_interface import Environment
# from algorithms.common.typing import Observation, Action, Value


class PPOTrainer:
    """
    Proximal Policy Optimization trainer (network-agnostic).

    Relies on the ActorCritic network to handle action sampling
    and log-probabilities.
    """

    def __init__(
        self,
        env: Environment,
        actor_critic: ActorCritic,
        rollout_length: int,
        ppo_epochs: int,
        mini_batch_size: int,
        gamma: float,
        lam: float,
        clip_eps: float,
        lr: float,
        device: torch.device,
    ):
        self.env = env
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

        self.rollout_length = rollout_length
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.B = env.batch_size
        self.obs_shape = env.obs_shape

        # Buffer
        self.buffer = PPORolloutBuffer(
            rollout_length=rollout_length,
            batch_size=self.B,
            obs_shape=self.obs_shape,
            device=device
        )

        # Initialize environment state
        self.obs = self.env.reset().to(device)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollout(self) -> None:
        """Collect a full rollout of length T into the buffer."""
        self.buffer.clear()

        for t in range(self.rollout_length):
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.act(self.obs)

            next_obs, reward, done = self.env.step(action)
            next_obs = next_obs.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
            value = value.to(self.device)
            log_prob = log_prob.to(self.device)

            self.buffer.add(self.obs, action, reward, done, value, log_prob)
            self.obs = next_obs

        # Bootstrap for last timestep
        with torch.no_grad():
            _, last_value = self.actor_critic.forward(self.obs)

        self.buffer.compute_returns_and_advantages(self.gamma, self.lam, last_value)

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self) -> tuple[float, float, float]:
        """
        Perform PPO update over collected rollout.

        Returns:
            (total_loss, policy_loss, value_loss)
        """
        total_loss = 0.0
        policy_loss_val = 0.0
        value_loss_val = 0.0

        dataloader = DataLoader(self.buffer, batch_size=self.mini_batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for batch in dataloader:
                b_obs, b_action, b_old_log_probs, b_advantages, b_returns = batch

                # Network handles action evaluation internally
                log_probs, entropy, values = self.actor_critic.evaluate_action(b_obs, b_action)

                # PPO surrogate
                ratio = (log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, b_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_val += policy_loss.item()
                value_loss_val += value_loss.item()

        # Average over epochs and mini-batches
        num_batches = self.ppo_epochs * len(dataloader)
        return total_loss / num_batches, policy_loss_val / num_batches, value_loss_val / num_batches

    # ------------------------------------------------------------------
    # Step method for one iteration
    # ------------------------------------------------------------------
    def step(self) -> tuple[float, float, float]:
        """Collect rollout and perform PPO update."""
        self.collect_rollout()
        return self.update()
