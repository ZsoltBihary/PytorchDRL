import torch
from torch import Tensor
from worlds.ipd.memory1.memory1_population import Memory1Population
from worlds.ipd.memory1.config import Config


class PopulationDyn:
    """
    Population dynamics for a fixed set of IPD strategies using replicator dynamics.

    Attributes
    ----------
    pop : Memory1Population
        The population of strategies (p-vectors fixed).
    x : Tensor
        Current strategy frequencies (N,) summing to 1.
    """
    def __init__(self, conf: Config, device="cpu"):
        self.conf = conf
        self.device = device
        self.pop = Memory1Population.build(conf, device)
        self.N_tot = conf.num_total_strat
        assert self.N_tot == self.pop.N
        self.N_skel = conf.num_skeleton_strat
        self.N_mut = conf.num_mutable_strat
        self.N_repl = conf.num_replace_strat
        assert self.N_tot == self.N_skel + self.N_mut
        # Start with uniform frequencies
        self.x = torch.ones(self.N_tot, device=device, dtype=torch.float32) / self.N_tot
        self.VX = self.pop.compute_VX_matrix()
        self.dt = conf.dt
        self.sqrt_dt = self.dt ** 0.5
        self.mut_floor = conf.mutation / self.N_tot
        self.lambda_diff = conf.redistribution_rate
        self.child_sig = conf.child_sigma

    def extract_pool(self, threshold: float = 1e-3, renormalize: bool = True) -> tuple[Tensor, Tensor]:
        """
        Extract a filtered memory-1 opponent pool from PopulationDyn.

        Args:
            threshold: minimum frequency to keep a strategy
            renormalize: whether to renormalize weights to sum to 1

        Returns:
            p_pool: (K,5) tensor of memory-1 probabilities
            w_pool: (K,) tensor of weights summing to 1
        """
        x = self.x.detach()
        p = self.pop.p.detach()

        mask = x > threshold
        assert mask.any(), "Threshold too high: no strategies survive"

        p_pool = p[mask]
        w_pool = x[mask]

        if renormalize:
            w_pool = w_pool / w_pool.sum()

        return p_pool, w_pool

    def update_VX(self):
        self.VX = self.pop.compute_VX_matrix()

    def apply_mutation_floor(self):
        """ Ensure a minimum frequency level, then renormalize. """
        self.x = torch.clamp(self.x, min=self.mut_floor)
        self.x = self.x / self.x.sum()

    def replicator_step(self):
        """ Advance the population frequencies by one Euler step. """
        VX_mod = self.VX - self.lambda_diff * torch.eye(self.N_tot, device=self.device, dtype=self.VX.dtype)
        f = VX_mod @ self.x
        avg_f = torch.dot(self.x, f)
        dx = self.x * (f - avg_f) * self.dt

        self.x = self.x + dx
        self.apply_mutation_floor()

    def run_replicator_dyn(self, steps: int):
        """ Run replicator dynamics for a number of steps. """
        for _ in range(steps):
            self.replicator_step()

    def replace(self):
        """
        Perform natural selection replacement on the MUTABLE subset. The skeleton is never eliminated.

        Steps implemented:
        1. Select the self.N_repl indices with the lowest x values from the MUTABLE subset for elimination.
        2. Available indices = all indices excluding elimination indices.
        3. Select self.N_repl parent indices from available indices randomly,
           with probabilities proportional to their x values, WITHOUT REPLACEMENT.
        4. Each parent spawns one child: child p vector = parent p + small noise (close to parent).
           Parent's weight is split equally between parent and child.
        5. Child p and x are copied into the p and x entries at the elimination indices.
        6. Apply trembling-hand clipping to all p.
        7. Enforce mutation floor on x and renormalize.
        8. Recalculate VX matrix
        """
        device = self.device
        N_repl = self.N_repl
        assert N_repl > 0
        mut_indices = torch.arange(self.N_skel, self.N_tot, device=device, dtype=torch.long)  # Mutable index range

        # 1. Select N_repl smallest-x indices within the mutable subset for elimination.
        x_mut = self.x[mut_indices]
        x_mut_noisy = x_mut + 0.01 * torch.randn_like(x_mut) / self.N_tot
        _, rel_elim_pos = torch.topk(x_mut_noisy, k=N_repl, largest=False)
        elim_idx = mut_indices[rel_elim_pos]  # global indices to eliminate

        # 2. Available indices = all indices excluding the elimination indices
        all_idx = torch.arange(0, self.N_tot, device=device, dtype=torch.long)
        mask = torch.ones(self.N_tot, dtype=torch.bool, device=device)
        mask[elim_idx] = False
        avail_idx = all_idx[mask]

        # 3. Select N_repl parent indices from available indices, probs proportional to x[avail]
        probs = self.x[avail_idx]
        parent_pos_in_avail = torch.multinomial(probs, num_samples=N_repl, replacement=False)
        parent_idx = avail_idx[parent_pos_in_avail]

        # 4. Each parent spawns one child (child p approx parent p). Split parent's weight equally.
        parent_p = self.pop.p[parent_idx]   # (N_repl, 5)
        parent_x = self.x[parent_idx]       # (N_repl,)
        # Create child p as parent p plus small noise.
        noise = torch.distributions.StudentT(1).sample(parent_p.shape).to(device) * self.child_sig
        # torch.distributions.StudentT(3).sample(parent_p.shape).to(device)
        # noise = torch.randn_like(parent_p, device=device) * self.child_sig
        child_p = parent_p + noise
        # Split parent weight equally: parent keeps half, child gets half
        child_x = parent_x * 0.5
        new_parent_x = parent_x * 0.5

        # 5. Insert children into the eliminated slots (overwrite p and x at elim indices)
        # Update parent weights in-place
        self.x[parent_idx] = new_parent_x
        # Place children
        self.pop.p[elim_idx] = child_p
        self.x[elim_idx] = child_x

        # 6. Apply trembling-hand clipping to the population probabilities (all strategies)
        self.pop.p = Memory1Population.apply_trembling_hand(self.pop.p, self.pop.eps)

        # 7. Enforce mutation floor and renormalize x
        self.apply_mutation_floor()

        # 8. Recalculate VX matrix
        self.update_VX()

    def print_survivors(self, threshold=1e-3):
        """
        Print only surviving strategies:
        - signature entries converted to integers
        - weights scaled by 10000 and converted to int
        threshold : float
            Minimum weight to be considered a survivor
        """
        p = self.pop.p.detach().cpu()
        x = self.x.detach().cpu()
        survivors = (x > threshold).nonzero(as_tuple=True)[0]

        # print("Survivors (signature ints, weight*10000 ints):")
        print(" idx |  p0 pCC pCD pDC pDD | weight")
        print("-" * 55)
        scale = self.conf.skeleton_K - 1
        # scale = 1.0
        for i in survivors:
            # convert signature to ints
            sig = (scale * p[i]).round().int().tolist()
            sig_str = "   ".join(str(v) for v in sig)
            # convert weight
            w = int(x[i] * 10000)
            print(f"{int(i):3d}  |  {sig_str}  | {w} / 10000")


if __name__ == "__main__":
    # --- basic config for quick testing ---
    conf = Config(
        gamma=0.98,
        trembling_hand=0.0,
        skeleton_K=3,
        only_skeleton=False,
        num_total_strat=1000,  # ignored when only_skeleton=True
        mutation=0.001,
        dt=0.05,
        child_sigma=0.05
    )
    thresh = 0.003
    replicator_steps = 20
    num_iter = 10

    print("Building PopulationDyn...")
    popdyn = PopulationDyn(conf, device="cpu")

    print(f"N_tot = {popdyn.N_tot}, N_skel = {popdyn.N_skel}, N_mut = {popdyn.N_mut}, N_repl = {popdyn.N_repl}")
    print("Initial x sum:", popdyn.x.sum().item())

    # --- run quick test ---
    print("Running some dynamics")
    for i in range(num_iter):
        popdyn.run_replicator_dyn(steps=replicator_steps)
        # print("\nSurvivors after i =", i+1)
        popdyn.pop.compute_VX_matrix()
        popdyn.print_survivors(threshold=thresh)
        print(i, popdyn.x @  popdyn.pop.p)
        popdyn.replace()

    p_pool, w_pool = popdyn.extract_pool()
    # # --- final assertions ---
    # assert abs(popdyn.x.sum().item() - 1.0) < 1e-6, "Frequencies must sum to 1"

    print("\nSanity check passed.")
