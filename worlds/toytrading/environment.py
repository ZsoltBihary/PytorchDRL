# worlds/toytrading/environment.py

import torch
from torch import Tensor
from drl.common.interfaces import Environment
from drl.common.types import Observation, Action, Reward, Done
from worlds.toytrading.config import Config, action_2_pos
from collections import namedtuple
import drl.common.tensor_tree as tt


class ToyTrading(Environment):
    def __init__(self, batch_size: int, gamma: float, random_termination: bool, conf: Config):

        # class Environment:
        #     def __init__(self, conf: Config):
        self.conf = conf
        # ===== Consume configuration parameters =====
        # === Data tensor sizes
        self._B = batch_size
        self._T = conf.window_size
        self._gamma = gamma
        self._rand_term = random_termination
        self._num_actions = 3
        # === Price dynamics
        self.S_mean = conf.S_mean
        self.vol = conf.volatility
        self.kappa = conf.mean_reversion
        # === Reward specification
        self.half_ba = conf.half_bidask
        self.risk_av = conf.risk_aversion
        self.device = None

        # Set up _obs_template
        Market = namedtuple("Market", ["price", "pos"])
        self._obs_template = Market(
                price=torch.zeros(self._T, dtype=torch.float32),
                pos=torch.tensor(0.0, dtype=torch.float32),
            )
        print("\n=== OBS TEMPLATE SUMMARY ===")
        tt.summary_tree(self._obs_template)

        # Set up environment internal state
        self.state = tt.batch_zeros(self._obs_template, batch_size=(self._B,))
        print("\n=== STATE PRINT ===")
        tt.print_tree(self.state, name="state")

        self.reset()
        print("\n=== STATE PRINT after reset ===")
        tt.print_tree(self.state, name="state")

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
    def obs_template(self) -> Observation:
        return self._obs_template

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _simulate_next_price(self, current_price: Tensor) -> Tensor:
        """Simulate one-step mean-reverting price update."""
        drift = -self.kappa * (current_price - self.S_mean)
        noise = self.vol * torch.randn_like(current_price, device=current_price.device)
        next_price = current_price + drift + noise
        return next_price

    def _simulate_initial_price_seq(self, n: int):
        """Simulate an initial window of prices via recursive mean reversion."""
        price_seq = torch.ones((n, self._T), dtype=torch.float32) * self.S_mean * 1.0
        for t in range(self._T - 1):
            price_seq[:, t+1] = self._simulate_next_price(price_seq[:, t])
        return price_seq

    # ---------------------------------------------------
    # core methods
    # ---------------------------------------------------
    def reset_state(self, mask: Tensor) -> None:
        """
        Reset selected environments.
        """
        if not mask.any():
            return

        # ===== Initialize prices
        n: int = mask.sum().item()
        self.state.price[mask, :] = self._simulate_initial_price_seq(n)
        self.state.pos[mask] = 0.0

    def apply(self, action: Action) -> tuple[Reward, Done]:
        # === Interpret action, simulate new price, compute reward
        new_pos = action_2_pos(action)
        new_price = self._simulate_next_price(self.state.price[:, -1])
        d_price = new_price - self.state.price[:, -1]
        reward = new_pos * d_price                                        # daily return
        reward -= self.half_ba * torch.abs(new_pos - self.state.pos)      # bid-ask friction
        reward -= self.risk_av * ((self.vol * new_pos) ** 2.0)            # risk aversion
        # === Compute *functional* new state = (new_price_seq, new_pos)
        new_price_seq = torch.roll(self.state.price, shifts=-1, dims=1)
        new_price_seq[:, -1] = new_price
        # === in-place commit
        self.state.price.copy_(new_price_seq)
        self.state.pos.copy_(new_pos)
        done = torch.zeros(self._B, device=self.device)
        return reward, done

    def get_obs(self) -> Observation:
        return tt.clone(self.state)


# import torch
# from torch import Tensor
# from src.config import Config, State, Action, Reward, action_2_pos
#
#
# class Environment:
#     def __init__(self, conf: Config):
#         self.conf = conf
#         # ===== Consume configuration parameters =====
#         # === Data tensor sizes
#         self.B = conf.batch_size
#         self.T = conf.window_size
#         # === Price dynamics
#         self.S_mean = conf.S_mean
#         self.vol = conf.volatility
#         self.kappa = conf.mean_reversion
#         # === Reward specification
#         self.half_ba = conf.half_bidask
#         self.risk_av = conf.risk_aversion
#
#         # ===== Initialize state tensors
#         self.price_seq = self._simulate_initial_price_seq()
#         self.pos = torch.zeros(self.B, dtype=torch.float32)
#
#     def _simulate_next_price(self, current_price: Tensor) -> Tensor:
#         """Simulate one-step mean-reverting price update."""
#         drift = -self.kappa * (current_price - self.S_mean)
#         noise = self.vol * torch.randn(self.B, device=current_price.device)
#         next_price = current_price + drift + noise
#         return next_price
#
#     def _simulate_initial_price_seq(self):
#         """Simulate an initial window of prices via recursive mean reversion."""
#         price_seq = torch.ones((self.B, self.T), dtype=torch.float32) * self.S_mean
#         for t in range(self.T - 1):
#             price_seq[:, t+1] = self._simulate_next_price(price_seq[:, t])
#         return price_seq
#
#     def get_state(self) -> State:
#         return (
#             self.price_seq.detach().clone(),
#             self.pos.detach().clone(),
#         )
#
#     def reset(self):
#         """Optional: reinitialize environment."""
#         self.price_seq = self._simulate_initial_price_seq()
#         self.pos = torch.zeros(self.B)
#         return self.get_state()
#
#     @torch.no_grad()
#     def step(self, action: Action) -> tuple[State, Reward]:
#         """
#         Advance the environment one time-step given an action.
#         Args:
#             action: long Tensor (B,)
#         Returns:
#             (new_price_seq, new_pos_seq): float32 tensors (B, T)
#             reward: float32 tensor (B,)
#         """
#         # === Interpret action, simulate new price, compute reward
#         new_pos = action_2_pos(action)
#         new_price = self._simulate_next_price(self.price_seq[:, -1])
#         d_price = new_price - self.price_seq[:, -1]
#         reward = new_pos * d_price                                  # daily return
#         reward -= self.half_ba * torch.abs(new_pos - self.pos)      # bid-ask friction
#         reward -= self.risk_av * ((self.vol * new_pos) ** 2.0)      # risk aversion
#         # === Compute *functional* new state = (new_price_seq, new_pos)
#         new_price_seq = torch.roll(self.price_seq, shifts=-1, dims=1)
#         new_price_seq[:, -1] = new_price
#         # new_pos = self.pos
#         # new_pos_seq[:, -1] = new_pos
#         # === in-place commit
#         self.price_seq.copy_(new_price_seq)
#         self.pos.copy_(new_pos)
#         # === return safe, detached output tensors
#         state = new_price_seq.detach(), new_pos.detach()
#         return state, reward.detach()
#
#     def print_data(self, price_seq, pos, n=3, reward=None):
#         n = min(n, self.B)
#         for i in range(n):
#             prices = price_seq[i].cpu().numpy()
#             # positions = pos_seq[i].cpu().numpy().astype(int)
#             position = pos[i].cpu().numpy()
#             print(f"[Agent {i:02d}] Prices: {prices.round(2)}")
#             print(f"           Pos   : {position}")
#             if reward is not None:
#                 rew = reward[i].cpu().numpy()
#                 print(f"           Reward : {float(rew):.2f}")
#             print("-" * 60)
#
#     def print_env(self, n=3):
#         """
#         Print a compact view of the internal class state.
#         Args:
#             n (int): Max number of batch elements to display (default: 3)
#         """
#         n = min(n, self.B)
#         print("=" * 60)
#         print(f"Environment snapshot (showing {n}/{self.B} agents):")
#         print(f"vol={self.vol:.4f}, kappa={self.kappa:.4f}, mean={self.S_mean:.2f}")
#         print("-" * 60)
#         self.print_data(price_seq=self.price_seq, pos=self.pos, n=n)
#
#
# # ------------------ SANITY CHECK ------------------ #
# if __name__ == "__main__":
#     conf = Config(window_size=10)
#     market = Environment(conf)
#     n_show = 2
#     print("\n ***** Initial state of market environment:")
#     market.print_env(n=n_show)
#     action = 0 * torch.ones(conf.batch_size, dtype=torch.long)  # Short
#     ((price_seq1, pos_seq1), reward1) = market.step(action)
#     action = 2 * torch.ones(conf.batch_size, dtype=torch.long)  # Long
#     ((price_seq2, pos_seq2), reward2) = market.step(action)
#     print("\n ***** Final state of market environment:")
#     market.print_env(n=n_show)
#     print("\n ***** Last output of resolve_action():")
#     market.print_data(price_seq2, pos_seq2, n=n_show, reward=reward2)
#
#     print("Sanity check passed.")
