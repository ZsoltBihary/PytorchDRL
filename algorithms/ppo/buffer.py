# algorithms/ppo/buffer.py

import torch
from torch.utils.data import Dataset
from algorithms.common.typing import (
    Observation,
    Action,
    Reward,
    Done,
    Value)
from algorithms.ppo.typing import LogProb


class PPORolloutBuffer(Dataset):
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
            raise RuntimeError("PPORolloutBuffer overflow")

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
