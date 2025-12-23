# drl/trainers/ppo.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from drl.common.types import Observation, Action, Reward, Done, Value, LogProb
from drl.common.interfaces import Environment
from drl.agents.policy_value_agent import PolicyValueAgent


class PPOTrainer:

    def __init__(self, env: Environment, agent: PolicyValueAgent, *,
                 rollout_length: int, lam: float,
                 epochs: int, mini_batch: int,
                 clip_eps: float, lr: float,
                 ent_coef: float = 0.01, max_grad_norm: float = 0.5):
        # consume parameters
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.gamma = env.gamma  # has to be consistent
        self.lam = lam
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        # buffer
        self.buffer = PPOBuffer(rollout_length=rollout_length, batch_size=env.batch_size,
                                obs_example=env.reset())
        # optimizer
        self.optimizer = torch.optim.Adam(self.agent.model.parameters(), lr=lr)
        # initialize observation
        self.obs = env.reset()

    # ---------------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------------
    @torch.no_grad()
    def rollout(self) -> float:
        self.buffer.clear()
        total_reward = 0.0
        for _ in range(self.rollout_length):
            # agent rollout call
            action, logits, value = self.agent.rollout_step(self.obs)
            # compute required PPO intermediates
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            # environment transition
            next_obs, reward, done = self.env.step(action)
            # push to buffer
            self.buffer.add(obs=self.obs, action=action, reward=reward, done=done,
                            value=value, log_prob=log_prob)

            self.obs = next_obs
            total_reward += reward.sum().item()
        mean_reward = total_reward / (self.rollout_length * self.env.batch_size)
        return mean_reward

    # ---------------------------------------------------------------
    # Update
    # ---------------------------------------------------------------
    def update(self):
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        dataloader = DataLoader(self.buffer, batch_size=self.mini_batch, shuffle=True)

        for _ in range(self.epochs):
            for b_obs, b_act, b_old_logp, b_adv, b_ret in dataloader:

                logits, values = self.agent.model(b_obs)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_act)
                entropy = dist.entropy()

                ratio = (log_probs - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - b_ret).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()

        num_batches = self.epochs * len(dataloader)

        return (
            total_loss / num_batches,
            policy_loss_sum / num_batches,
            value_loss_sum / num_batches,
        )

    # ---------------------------------------------------------------
    # Trainer step: rollout → bootstrap/GAE → update
    # ---------------------------------------------------------------
    def step(self):
        # rollout
        self.agent.model.eval()
        mean_reward = self.rollout()
        # bootstrap + compute GAE
        with torch.no_grad():
            _, _, last_val = self.agent.rollout_step(self.obs)

        self.buffer.compute_returns_and_advantages(gamma=self.gamma, lam=self.lam,
                                                   last_value=last_val)
        # model update
        self.agent.model.train()
        total, pol, val = self.update()

        return dict(
            mean_reward=mean_reward,
            total_loss=total,
            policy_loss=pol,
            value_loss=val,
        )


# ------------------------------------------------------------------
# PPO Buffer
# ------------------------------------------------------------------
class PPOBuffer(Dataset):
    """
    PPO rollout buffer with static (T, B, ...) tensor storage.
    Supports recursive Observation trees.
    """
    def __init__(
        self,
        rollout_length: int,
        batch_size: int,
        obs_example: Observation,  # example observation to define storage shape
        dtype: torch.dtype = torch.float32,
    ):
        self.T = rollout_length
        self.B = batch_size
        self.dtype = dtype

        # Allocate storage recursively
        self.observations: Observation = self._allocate_obs_storage(obs_example, self.T, self.B, self.dtype)
        self.actions = torch.zeros((self.T, self.B), dtype=torch.long)
        self.rewards = torch.zeros((self.T, self.B), dtype=self.dtype)
        self.dones = torch.zeros((self.T, self.B), dtype=self.dtype)
        self.values = torch.zeros((self.T, self.B), dtype=self.dtype)
        self.log_probs = torch.zeros((self.T, self.B), dtype=self.dtype)
        self.advantages = torch.zeros((self.T, self.B), dtype=self.dtype)
        self.returns = torch.zeros((self.T, self.B), dtype=self.dtype)

        self.ptr = 0

    # ------------------------------------------------------------------
    # RolloutBuffer interface
    # ------------------------------------------------------------------
    def clear(self) -> None:
        self.ptr = 0

    def add(self, obs: Observation, action: Action, reward: Reward, done: Done,
            value: Value, log_prob: LogProb) -> None:
        """Add one timestep of batched data to the buffer."""
        if self.ptr >= self.T:
            raise RuntimeError("PPOBuffer overflow")

        self._write_obs(self.observations, obs, self.ptr)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob

        self.ptr += 1

    def compute_returns_and_advantages(self, gamma: float, lam: float, last_value: Value) -> None:
        """Compute GAE advantages and returns."""
        if self.ptr != self.T:
            raise RuntimeError("Cannot compute GAE: buffer not full")

        gae = torch.zeros(self.B, dtype=self.dtype)

        for t in reversed(range(self.T)):
            next_value = last_value if t == self.T - 1 else self.values[t + 1]
            next_done = self.dones[t] if t == self.T - 1 else self.dones[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1.0 - next_done) - self.values[t]
            gae = delta + gamma * lam * (1.0 - next_done) * gae
            self.advantages[t] = gae

        self.returns.copy_(self.advantages + self.values)

    # ------------------------------------------------------------------
    # Dataset interface (flattened view)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.T * self.B

    def __getitem__(self, index: int):
        t = index // self.B
        b = index % self.B
        return (
            self._read_obs(self.observations, t, b),
            self.actions[t, b],
            self.log_probs[t, b],
            self.advantages[t, b],
            self.returns[t, b],
        )

    # ------------------------------------------------------------------
    # Recursive utility functions
    # ------------------------------------------------------------------
    def _allocate_obs_storage(self, spec: Observation, T: int, B: int, dtype: torch.dtype) -> Observation:
        """Recursively allocate a storage structure mirroring the observation tree."""
        if isinstance(spec, torch.Tensor):
            shape = spec.shape[1:]  # leave out batch dim
            return torch.zeros((T, B, *shape), dtype=dtype)
        elif isinstance(spec, tuple):
            return tuple(self._allocate_obs_storage(s, T, B, dtype) for s in spec)
        else:
            raise TypeError(f"Unsupported spec type: {type(spec)}")

    def _write_obs(self, storage: Observation, obs: Observation, t: int) -> None:
        """Recursively write a batch of observations at timestep t."""
        if isinstance(storage, torch.Tensor):
            storage[t] = obs
        elif isinstance(storage, tuple):
            for s_child, o_child in zip(storage, obs):
                self._write_obs(s_child, o_child, t)
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")

    def _read_obs(self, storage: Observation, t: int, b: int) -> Observation:
        """Recursively read a single flattened observation."""
        if isinstance(storage, torch.Tensor):
            return storage[t, b]
        elif isinstance(storage, tuple):
            return tuple(self._read_obs(s, t, b) for s in storage)
        else:
            raise TypeError(f"Unsupported storage type: {type(storage)}")
