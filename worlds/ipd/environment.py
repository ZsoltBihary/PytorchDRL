# worlds/ipd/environment.py

import torch
from torch import Tensor
from drl.common.interfaces import Environment
from drl.common.types import Observation, Action, Reward, Done


class IteratedPrisonersDilemma(Environment):
    """
    Vectorized Iterated Prisoner's Dilemma with a population of memory-1 opponents.
    """

    def __init__(
        self,
        batch_size: int,
        history_len: int,
        opponent_probs: Tensor,
        opponent_weights: Tensor,
        gamma: float = 0.95,
        device: torch.device | None = None,
    ):
        """
        Args:
            batch_size: number of parallel environments
            history_len: L (agent memory length)
            opponent_probs: Tensor (K, 5) memory-1 cooperate probabilities
            opponent_weights: Tensor (K,) weights summing to 1
            gamma: discount factor (used for random termination)
        """
        self._B = batch_size
        self.L = history_len
        self.device = device or opponent_probs.device

        # opponent pool
        self.opponent_probs = opponent_probs.to(self.device)      # (K, 5)
        self.opponent_weights = opponent_weights.to(self.device)  # (K,)

        self._gamma = gamma

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
        return True

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.L, 5

    @property
    def num_actions(self) -> int:
        return 2  # D or C

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
        self.last_agent_action[mask] = 0
        self.last_opponent_action[mask] = 0

    def apply(self, action: Action) -> tuple[Reward, Done]:
        """
        Execute one IPD step.
        """
        B = self.batch_size
        device = self.device

        # ---------------------------------------------
        # determine memory-1 state index
        # ---------------------------------------------
        # if history is empty → state = 0
        prev_state = torch.argmax(self.history[:, -1], dim=-1)  # (B,)

        # opponent cooperate probability
        opp_p = self.opponent_probs[self.opponent_idx, prev_state]
        opp_action = (torch.rand(B, device=device) < opp_p).long()

        # ---------------------------------------------
        # rewards
        # ---------------------------------------------
        # payoff matrix
        # agent_action, opp_action → reward
        reward = torch.zeros(B, device=device)

        reward[(action == 1) & (opp_action == 1)] = 3.0
        reward[(action == 1) & (opp_action == 0)] = 0.0
        reward[(action == 0) & (opp_action == 1)] = 5.0
        reward[(action == 0) & (opp_action == 0)] = 1.0

        # ---------------------------------------------
        # update history
        # ---------------------------------------------
        new_state = torch.zeros(B, device=device, dtype=torch.long)

        # CC = 1, CD = 2, DC = 3, DD = 4
        new_state[(action == 1) & (opp_action == 1)] = 1
        new_state[(action == 1) & (opp_action == 0)] = 2
        new_state[(action == 0) & (opp_action == 1)] = 3
        new_state[(action == 0) & (opp_action == 0)] = 4

        # shift history
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = 0.0
        self.history[torch.arange(B), -1, new_state] = 1.0

        # update last actions
        self.last_agent_action = action
        self.last_opponent_action = opp_action

        done = torch.zeros(B, device=device)  # handled by random termination
        return reward, done

    def get_obs(self) -> Observation:
        return self.history
