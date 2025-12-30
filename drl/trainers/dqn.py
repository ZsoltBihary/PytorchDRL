# drl/trainers/dqn.py

import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from drl.common.types import Observation, Action, Reward, Done
from drl.common.interfaces import Environment
from drl.agents.q_value_agent import QValueAgent


class DQNTrainer:
    """
    DQN Trainer (batched rollout + replay, target network)
    """

    def __init__(
        self,
        env: Environment,
        agent: QValueAgent,
        *,
        rollout_length: int,
        buffer_capacity: int,
        epochs: int,
        mini_batch: int,
        lr: float,
        max_grad_norm: float | None = None,
    ):
        self.env = env
        self.agent = agent
        self.gamma = env.gamma

        self.rollout_length = rollout_length
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.max_grad_norm = max_grad_norm

        # Replay buffer
        self.buffer = DQNBuffer(
            capacity=buffer_capacity,
            batch_size=env.batch_size,
            obs_example=env.reset(),
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.model.parameters(),
            lr=lr,
        )

        # Target network
        self.target_model = deepcopy(self.agent.model)
        self.target_model.eval()
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Initial observation
        self.obs = env.reset()

    # ------------------------------------------------------------
    # Rollout phase: collect multiple env steps
    # ------------------------------------------------------------
    @torch.no_grad()
    def rollout(self) -> float:
        total_reward = 0.0
        self.agent.model.eval()

        for _ in range(self.rollout_length):
            action, _ = self.agent.rollout_step(self.obs)
            next_obs, reward, done = self.env.step(action)

            self.buffer.add(
                obs=self.obs,
                action=action,
                reward=reward,
                done=done,
                next_obs=next_obs,
            )

            self.obs = next_obs
            total_reward += reward.sum().item()

        mean_reward = total_reward / (self.rollout_length * self.env.batch_size)
        return mean_reward

    # ------------------------------------------------------------
    # Update phase: full randomized scan over buffer
    # ------------------------------------------------------------
    def update(self) -> float:
        if len(self.buffer) < self.mini_batch:
            return 0.0

        dataloader = DataLoader(
            self.buffer,
            batch_size=self.mini_batch,
            shuffle=True,       # full randomized scan
        )

        total_loss = 0.0
        num_batches = 0
        self.agent.model.train()

        for _ in range(self.epochs):
            for b_obs, b_act, b_rew, b_done, b_next_obs in dataloader:

                # TD target
                with torch.no_grad():
                    q_next = self.target_model(b_next_obs)
                    max_q_next = q_next.max(dim=-1).values
                    target = b_rew + (1.0 - b_done) * self.gamma * max_q_next

                # Online Q-values
                q = self.agent.model(b_obs)
                q_a = q.gather(1, b_act.unsqueeze(1)).squeeze(1)

                loss = F.mse_loss(q_a, target)

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(),
                        self.max_grad_norm,
                    )
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    # ------------------------------------------------------------
    # One training step: rollout → update → target sync
    # ------------------------------------------------------------
    def step(self) -> dict:
        mean_reward = self.rollout()
        loss = self.update()

        # Slow timescale target update
        self.target_model.load_state_dict(self.agent.model.state_dict())

        return dict(
            mean_reward=mean_reward,
            loss=loss,
        )


class DQNBuffer(Dataset):
    """
    Circular replay buffer with batched add().
    - Stores flattened transitions (no time dimension)
    - Supports recursive Observation tensor trees
    - CPU-oriented (advanced indexing is fine)
    """

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        obs_example: Observation,
        dtype: torch.dtype = torch.float32,
    ):
        self.capacity = capacity
        self.B = batch_size
        self.dtype = dtype

        # Allocate storage
        self.observations = self._allocate_obs_storage(obs_example, capacity, dtype)
        self.next_observations = self._allocate_obs_storage(obs_example, capacity, dtype)

        self.actions = torch.zeros((capacity,), dtype=torch.long)
        self.rewards = torch.zeros((capacity,), dtype=dtype)
        self.dones = torch.zeros((capacity,), dtype=dtype)

        # Circular buffer state
        self.ptr = 0          # next write position
        self.size = 0         # number of valid transitions stored

    # ------------------------------------------------------------------
    # Buffer interface
    # ------------------------------------------------------------------
    def add(
        self,
        obs: Observation,
        action: Action,
        reward: Reward,
        done: Done,
        next_obs: Observation,
    ) -> None:
        """
        Add a batch of transitions.

        Args:
            obs:       Observation at time t, shape (B, ...)
            action:    Action taken, shape (B,)
            reward:    Reward received, shape (B,)
            done:      Done mask, shape (B,)
            next_obs:  Observation at time t+1, shape (B, ...)
        """
        B = action.shape[0]
        if B != self.B:
            raise ValueError(f"Expected batch size {self.B}, got {B}")

        device = self.actions.device
        idx = (self.ptr + torch.arange(B, device=device)) % self.capacity

        # Scalar fields
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done

        # Observation trees
        self._write_obs_batch(self.observations, obs, idx)
        self._write_obs_batch(self.next_observations, next_obs, idx)

        # Advance circular pointer
        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        if index >= self.size:
            raise IndexError

        return (
            self._read_obs(self.observations, index),
            self.actions[index],
            self.rewards[index],
            self.dones[index],
            self._read_obs(self.next_observations, index),
        )

    # ------------------------------------------------------------------
    # Recursive observation utilities (mirrors PPOBuffer style)
    # ------------------------------------------------------------------
    def _allocate_obs_storage(
        self,
        spec: Observation,
        N: int,
        dtype: torch.dtype,
    ) -> Observation:
        """Recursively allocate observation storage."""
        if isinstance(spec, torch.Tensor):
            shape = spec.shape[1:]  # strip batch dim
            return torch.zeros((N, *shape), dtype=dtype)
        elif isinstance(spec, tuple):
            return tuple(self._allocate_obs_storage(s, N, dtype) for s in spec)
        else:
            raise TypeError(f"Unsupported spec type: {type(spec)}")

    def _write_obs_batch(
        self,
        storage: Observation,
        obs: Observation,
        idx: torch.Tensor,
    ) -> None:
        """Recursively write a batch of observations at indices idx."""
        if isinstance(storage, torch.Tensor):
            # storage: (N, *obs_template)
            # obs:     (B, *obs_template)
            storage[idx] = obs
        elif isinstance(storage, tuple):
            for s_child, o_child in zip(storage, obs):
                self._write_obs_batch(s_child, o_child, idx)
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")

    def _read_obs(
        self,
        storage: Observation,
        idx: int,
    ) -> Observation:
        """Recursively read a single observation."""
        if isinstance(storage, torch.Tensor):
            return storage[idx]
        elif isinstance(storage, tuple):
            return tuple(self._read_obs(s, idx) for s in storage)
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")
