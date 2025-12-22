import torch
from torch import Tensor
import itertools
from config import Config


class StratPopulation:
    """
    A population of memory‑1 IPD strategies.

    Strategy representation (N players):
        p : (N, 5) tensor
            Columns: [p0, p_CC, p_CD, p_DC, p_DD]
            p0      : Probability to cooperate in the *first* round
            p_*     : Probabilities to cooperate given previous state
                        CC, CD, DC, DD

    Payoffs:
        payoffs : (4,) tensor = [R, S, T, P]
        R  = Reward        : payoff when both players Cooperate           CC
        S  = Sucker's      : payoff when you Cooperate, opponent Defects  CD
        T  = Temptation    : payoff when you Defect, opponent Cooperates  DC
        P  = Punishment    : payoff when both players Defect              DD
        These are X's payoffs; Y's payoffs are symmetric.

    Discount factor:
        gamma : float, 0 < gamma < 1

    Trembling hand noise:
        eps : float, 0 <= eps < 0.5
    """

    def __init__(self, conf: Config, p: Tensor):
        """
        Primary constructor, using configuration
        p : (N,5) tensor
        """
        self.device = p.device
        self.p = p.float()
        self.N = p.shape[0]
        # assert self.N == conf.num_total_strat

        self.gamma = conf.gamma
        self.payoffs = conf.payoffs.to(self.device)
        self.eps = conf.trembling_hand

    @staticmethod
    def apply_trembling_hand(p, eps):
        return torch.clamp(p, eps, 1.0 - eps)

    @staticmethod
    def make_K_grid(K, device="cpu"):
        G = torch.linspace(0., 1., K, device=device)
        grid = torch.tensor(list(itertools.product(G, repeat=5)), device=device, dtype=torch.float32)
        return grid

    @classmethod
    def build(cls, conf: Config, device="cpu"):
        """ Build a StratPopulation based on the configuration. """

        # 1. Skeleton grid (K^5 points)
        skeleton = cls.make_K_grid(conf.skeleton_K, device=device)
        # 2. Mutable random strategies (if needed)
        if conf.num_mutable_strat > 0:
            mutable = torch.rand(conf.num_mutable_strat, 5, device=device)
            p = torch.cat([skeleton, mutable], dim=0)
        else:
            p = skeleton
        assert p.shape[0] == conf.num_total_strat
        # 3. Apply trembling-hand clipping
        p = cls.apply_trembling_hand(p, conf.trembling_hand)
        # 4. Primary constructor call
        return cls(conf, p)

    # ------------------------------------------------------------------
    def compute_VX_matrix(self):
        """
        Compute VX(i,j) for all ordered pairs (i,j),
        forming an (N,N) matrix of discounted per-turn expected returns.

        Algorithm:
            For each ordered pair (i,j), the memory-1 IPD defines a 4-state
            Markov chain over states {CC, CD, DC, DD}. The transition matrix
            M is determined by p_mem[i] and p_mem[j]. The expected discounted
            occupancy vector z satisfies:

                (I - gamma * M^T) z = e0

            where e0 is the distribution over the *first* move outcomes.
            Then:  V_X = z · payoffs.

        Returns
        -------
        VX : (N,N) tensor
            VX[i,j] = per-round expected return for X=i vs Y=j
        """
        device = self.device
        gamma = self.gamma
        N = self.N

        # --- 1. Extract strategies ---
        p0 = self.p[:, 0]  # (N,)
        p_mem = self.p[:, 1:]  # (N,4) [p_CC, p_CD, p_DC, p_DD]

        # --- 2. Reorder Y's probabilities to X perspective ---
        y_row_map = torch.tensor([0, 2, 1, 3], device=device)  # CD<->DC swap
        pX = p_mem  # (N,4)
        pY = p_mem[:, y_row_map]  # (N,4)

        # --- 3. Broadcast to all pairs ---
        pX_exp = pX[:, None, :]  # (N,1,4)
        pY_exp = pY[None, :, :]  # (1,N,4)

        # --- 4. Construct transition matrix M (row=prev state, col=next state) ---
        CC = pX_exp * pY_exp
        CD = pX_exp * (1 - pY_exp)
        DC = (1 - pX_exp) * pY_exp
        DD = (1 - pX_exp) * (1 - pY_exp)

        M = torch.stack([CC, CD, DC, DD], dim=-1)  # (N,N,4,4)

        # --- 5. Solve (I - gamma * M^T) y = payoffs for each pair ---
        Id = torch.eye(4, device=device)
        payoffs_exp = self.payoffs[None, None, :]  # (1,1,4)
        y = torch.linalg.solve(Id - gamma * M, payoffs_exp.expand(N, N, -1))  # (N,N,4)

        # --- 6. Construct initial state vector e0 ---
        p0X = p0[:, None]  # (N,1)
        p0Y = p0[None, :]  # (1,N)
        e0 = torch.stack([
            p0X * p0Y,  # CC
            p0X * (1 - p0Y),  # CD
            (1 - p0X) * p0Y,  # DC
            (1 - p0X) * (1 - p0Y)  # DD
        ], dim=-1)  # (N,N,4)

        # --- 7. Compute per-round expected return ---
        VX = (1 - gamma) * (e0 * y).sum(dim=-1)  # (N,N)
        return VX


if __name__ == "__main__":
    # -------------------------------
    # Sanity check for StratPopulation
    # -------------------------------
    conf = Config(only_skeleton=True, skeleton_K=2)
    print("=== Config parameters ===")
    print(f"gamma: {conf.gamma}")
    print(f"trembling_hand: {conf.trembling_hand}")
    print(f"skeleton_K: {conf.skeleton_K}")
    print(f"num_total_strat: {conf.num_total_strat}")
    print(f"num_skeleton_strat: {conf.num_skeleton_strat}")
    print(f"num_mutable_strat: {conf.num_mutable_strat}")
    print(f"mutation: {conf.mutation}")
    print(f"payoffs: {conf.payoffs}")

    # Build the population
    pop = StratPopulation.build(conf)
    print("\n=== StratPopulation sanity check ===")
    print(f"p shape: {pop.p.shape}")
    print(f"p device: {pop.p.device}")
    print(f"min(p): {pop.p.min().item():.4f}")
    print(f"max(p): {pop.p.max().item():.4f}")

    # Print first few strategies
    print("\nFirst 5 strategies (p0, p_CC, p_CD, p_DC, p_DD):")
    for i in range(min(5, pop.p.shape[0])):
        print(f"{i}: {pop.p[i].tolist()}")

    VX_sel = pop.compute_VX_matrix()

    print("\nV_X matrix:")
    # print(VX_sel)
    print("Shape:", VX_sel.shape)

    avg_V = VX_sel.mean(dim=1)
    print("Shape:", avg_V.shape)
    print(avg_V)
