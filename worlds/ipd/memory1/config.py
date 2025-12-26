import torch


class Config:
    def __init__(
            self,
            # ===== Parameters =====
            gamma: float = 0.95,                    # discount factor
            trembling_hand: float = 0.05,            # execution noise
            skeleton_K: int = 2,                    # fixed skeleton grid dimension
            only_skeleton: bool = True,             # flag for only constructing the fixed skeleton
            num_total_strat: int = 32,            # total number of strategies
            mutation: float = 0.0,                  # mutation ratio
            # Nowak–Sigmund / Press–Dyson parametrization
            b_NS: float = 2.0,
            c_NS: float = 0.5,
            # b_NS = benefit of receiving cooperation
            # c_NS = cost of giving cooperation
            # b_NS > 1, c_NS > 0
            # [R, S, T, P] = [1, -c, b, 0] for plays [CC, CD, DC, DD]
            dt: float = 0.05,                       # time-step for replicator dynamics
            replace_ratio: float = 0.1,             # N_repl = replace_ratio * N_mut
            child_sigma: float = 0.0,
            redistribution_rate: float = 0.35,
    ):
        # ===== Parameters =====
        self.gamma = gamma
        self.trembling_hand = trembling_hand
        self.skeleton_K = skeleton_K
        self.num_skeleton_strat = skeleton_K ** 5
        if only_skeleton:
            self.num_total_strat = self.num_skeleton_strat
        else:
            self.num_total_strat = num_total_strat
        self.num_mutable_strat = self.num_total_strat - self.num_skeleton_strat
        self.num_replace_strat = int(self.num_mutable_strat * replace_ratio)

        assert self.num_mutable_strat >= 0, (
            f"num_total_strat={self.num_total_strat} must be >= skeleton_K^5 "
            f"={self.num_skeleton_strat}"
        )
        self.mutation = mutation
        # [R, S, T, P] = [1, -c, b, 0] for plays [CC, CD, DC, DD]
        self.payoffs = torch.tensor([1.0, -c_NS, b_NS, 0.0])
        self.dt = dt
        self.child_sigma = child_sigma
        self.redistribution_rate = redistribution_rate


if __name__ == "__main__":
    # --- basic config for quick testing ---
    conf = Config(
        gamma=0.98,
        trembling_hand=0.002,
        skeleton_K=3,
        only_skeleton=False,
        num_total_strat=1000,  # ignored when only_skeleton=True
        mutation=0.001,
        dt=0.02
    )

    a = 42
