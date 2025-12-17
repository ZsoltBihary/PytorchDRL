# drl/trainers/ppo.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from drl.common.types import Observation, Action, Reward, Done, Value, LogProb, Entropy
from drl.common.interfaces import Environment, PolicyValueModel


class PPOTrainer:
    """
    Refactored Proximal Policy Optimization trainer.

    Responsibilities:
    - Owns environment, model, optimizer, buffer
    - Collects rollouts with stochastic policy sampling
    - Computes log_probs, entropy, values for policy gradient
    - Performs PPO updates
    """

    def __init__(
        self,
        env: Environment,
        model: PolicyValueModel,
        rollout_length: int,
        ppo_epochs: int,
        mini_batch_size: int,
        gamma: float,
        lam: float,
        clip_eps: float,
        lr: float,
        device: torch.device,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.model = model
        self.device = device
        self.rollout_length = rollout_length
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        # Environment batch info
        self.B = env.batch_size
        self.obs_shape = env.obs_shape

        # Buffer
        self.buffer = PPOBuffer(
            rollout_length=rollout_length,
            batch_size=self.B,
            obs_shape=self.obs_shape,
            device=device
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Initial environment state
        self.obs = self.env.reset().to(self.device)

    # -------------------------------------------------
    # Rollout: sample actions with side effects
    # -------------------------------------------------
    @torch.no_grad()
    def _sample_action(self, obs: Observation) -> tuple[Action, LogProb, Value, Entropy]:
        """
        Sample actions from policy logits for training, compute log_probs, entropy, and value.
        No temperature scaling.
        """
        logits, value = self.model(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def collect_rollout(self) -> float:
        """
        Collect a full rollout into the buffer.
        Returns mean reward over rollout.
        """
        self.buffer.clear()
        total_reward = 0.0

        for t in range(self.rollout_length):
            action, log_prob, value, _ = self._sample_action(self.obs)
            next_obs, reward, done = self.env.step(action)

            # Move tensors to device
            next_obs = next_obs.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
            value = value.to(self.device)
            log_prob = log_prob.to(self.device)

            # Store in buffer
            self.buffer.add(self.obs, action, reward, done, value, log_prob)

            # Update state
            self.obs = next_obs
            total_reward += reward.sum().item()

        # Bootstrap last value
        with torch.no_grad():
            _, last_value = self.model(self.obs)
        self.buffer.compute_returns_and_advantages(self.gamma, self.lam, last_value)

        mean_reward = total_reward / (self.rollout_length * self.B)
        return mean_reward

    # -------------------------------------------------
    # PPO update
    # -------------------------------------------------
    def update(self) -> tuple[float, float, float]:
        """
        Perform PPO update over collected rollout.
        Returns total_loss, policy_loss, value_loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        policy_loss_sum = torch.tensor(0.0, device=self.device)
        value_loss_sum = torch.tensor(0.0, device=self.device)

        dataloader = DataLoader(self.buffer, batch_size=self.mini_batch_size, shuffle=True)

        for _ in range(self.ppo_epochs):
            for batch in dataloader:
                b_obs, b_action, b_old_log_probs, b_advantages, b_returns = batch

                b_obs = b_obs.to(self.device)
                b_action = b_action.to(self.device)
                b_old_log_probs = b_old_log_probs.to(self.device)
                b_advantages = b_advantages.to(self.device)
                b_returns = b_returns.to(self.device)

                logits, values = self.model(b_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_action)
                entropy = dist.entropy()

                # PPO surrogate loss
                ratio = (log_probs - b_old_log_probs).exp()
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, b_returns)
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy.mean()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss
                policy_loss_sum += policy_loss
                value_loss_sum += value_loss

        num_batches = self.ppo_epochs * len(dataloader)
        return (
            (total_loss / num_batches).detach().cpu().item(),
            (policy_loss_sum / num_batches).detach().cpu().item(),
            (value_loss_sum / num_batches).detach().cpu().item(),
        )

    # -------------------------------------------------
    # Step: one full PPO iteration
    # -------------------------------------------------
    def step(self) -> dict[str, float]:
        """
        Collect rollout and perform PPO update.
        Returns dict with metrics.
        """
        self.model.eval()
        mean_reward = self.collect_rollout()
        self.model.train()
        total_loss, policy_loss, value_loss = self.update()

        return {
            "mean_reward": mean_reward,
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }


class PPOBuffer(Dataset):
    """
    PPO rollout buffer with static (T, B, ...) tensor storage.

    Internally stores data in shape (T, B, ...).
    Exposes a flattened (T * B) Dataset interface for minibatching.
    """
    def __init__(
        self,
        rollout_length: int,
        batch_size: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.T = rollout_length
        self.B = batch_size
        self.obs_shape = obs_shape
        self.device = device
        self.dtype = dtype

        self._allocate_storage()
        self.ptr = 0

    # ------------------------------------------------------------------
    # Storage allocation
    # ------------------------------------------------------------------
    def _allocate_storage(self) -> None:
        """Allocate all tensors with fixed shapes."""
        self.observations = torch.zeros((self.T, self.B, *self.obs_shape), device=self.device, dtype=self.dtype)
        self.actions = torch.zeros((self.T, self.B), device=self.device, dtype=torch.long)
        self.rewards = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)
        self.dones = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)
        self.values = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)
        self.log_probs = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)
        self.advantages = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)
        self.returns = torch.zeros((self.T, self.B), device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # RolloutBuffer interface
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset write pointer; data will be overwritten."""
        self.ptr = 0

    def add(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        done: Done,
        value: Value,
        log_prob: LogProb,
    ) -> None:
        """Add one timestep of batched data to the buffer."""
        if self.ptr >= self.T:
            raise RuntimeError("PPOBuffer overflow")

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def compute_returns_and_advantages(self, gamma: float, lam: float, last_value: Value) -> None:
        """
        Compute GAE advantages and returns.

        Args:
            gamma: discount factor
            lam: GAE lambda
            last_value: value estimate for state after final timestep, shape (B,)
        """
        if self.ptr != self.T:
            raise RuntimeError("Cannot compute GAE: buffer not full")

        gae = torch.zeros(
            self.B, device=self.device, dtype=self.dtype
        )

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_value = last_value
                next_done = self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * (1.0 - next_done)
                - self.values[t]
            )

            gae = delta + gamma * lam * (1.0 - next_done) * gae
            self.advantages[t] = gae

        self.returns.copy_(self.advantages + self.values)

    # ------------------------------------------------------------------
    # Dataset interface (flattened view)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.T * self.B

    def __getitem__(self, index: int):
        """
        Return one flattened (t, b) sample.
        Mapping:
            t = index // B
            b = index % B
        """
        t = index // self.B
        b = index % self.B

        return (
            self.observations[t, b],
            self.actions[t, b],
            self.log_probs[t, b],
            self.advantages[t, b],
            self.returns[t, b],
        )
