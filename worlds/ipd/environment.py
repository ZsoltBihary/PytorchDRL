# worlds/ipd/environment.py

import torch
from torch import Tensor
from drl.common.interfaces import Environment
from drl.common.types import Observation, Action, Reward, Done


class IteratedPrisonersDilemma(Environment):
    """
    Vectorized Iterated Prisoner's Dilemma with a population of memory-1 opponents.
    """
    def __init__(self, batch_size: int, gamma: float, random_termination: bool,
                 history_len: int,
                 opponent_probs: Tensor,
                 opponent_weights: Tensor,
                 payoffs: Tensor,
                 device: torch.device | None = None):
        """
        Args:
            batch_size: number of parallel environments
            gamma: discount factor (used for random termination or discounting)
            random_termination: flag for random termination
            history_len: L (agent memory length)
            opponent_probs: Tensor (K, 5) memory-1 cooperate probabilities
            opponent_weights: Tensor (K,) weights summing to 1
        """
        # consume parameters
        self._B = batch_size
        self._gamma = gamma
        self._rand_term = random_termination
        self._L = history_len
        self._num_actions = 2
        self.device = device or opponent_probs.device

        # opponent pool
        self.opponent_probs = opponent_probs.to(self.device)      # (pool_size, 5)
        self.opponent_weights = opponent_weights.to(self.device)  # (pool_size,)
        self.payoffs = payoffs.to(self.device).float()            # (4,)
        assert self.payoffs.shape == (4,)
        self.agent_to_opp_state = torch.tensor(data=[0, 1, 3, 2, 4], device=self.device)

        # internal state
        self.history = torch.zeros(
            batch_size, history_len, 5, device=self.device
        )
        self.last_agent_action = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self.last_opponent_action = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        # sampled opponent index per env
        self.opponent_idx = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        self.reset()

    # ---------------------------------------------------
    # properties
    # ---------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self._B

    @property
    def random_termination(self) -> bool:
        return self._rand_term

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self._L, 5

    @property
    def num_actions(self) -> int:
        return self._num_actions  # = 2: D or C

    # ---------------------------------------------------
    # core methods
    # ---------------------------------------------------
    def reset_state(self, mask: Tensor) -> None:
        """
        Reset selected environments.
        """
        if not mask.any():
            return

        # sample opponents
        new_idx = torch.multinomial(
            self.opponent_weights,
            num_samples=mask.sum().item(),
            replacement=True,
        )
        self.opponent_idx[mask] = new_idx

        # clear history
        self.history[mask] = 0.0
        self.history[mask, :, 0] = 1.0  # Empty state

        # clear last actions
        # self.last_agent_action[mask] = 0
        # self.last_opponent_action[mask] = 0

    def apply(self, action: Action) -> tuple[Reward, Done]:
        B = self.batch_size
        device = self.device

        # --- opponent action ---
        agent_prev_state = torch.argmax(self.history[:, -1], dim=-1)
        opp_prev_state = self.agent_to_opp_state[agent_prev_state]
        opp_p = self.opponent_probs[self.opponent_idx, opp_prev_state]
        opp_action = (torch.rand(B, device=device) < opp_p).long()

        # --- payoff indexing ---
        # outcome index: 0=CC, 1=CD, 2=DC, 3=DD
        outcome = torch.zeros(B, dtype=torch.long, device=device)

        outcome[(action == 1) & (opp_action == 1)] = 0  # CC
        outcome[(action == 1) & (opp_action == 0)] = 1  # CD
        outcome[(action == 0) & (opp_action == 1)] = 2  # DC
        outcome[(action == 0) & (opp_action == 0)] = 3  # DD

        reward = self.payoffs[outcome]

        # --- update history (agent perspective) ---
        new_state = outcome + 1  # CC→1, CD→2, DC→3, DD→4

        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1, :] = 0.0
        self.history[torch.arange(B), -1, new_state] = 1.0
        done = torch.zeros(B, device=device)
        return reward, done

    def get_obs(self) -> Observation:
        return self.history
