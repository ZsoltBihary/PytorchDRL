# import torch
from config import Config
# from strat_population import StratPopulation
from population_dyn import PopulationDyn

if __name__ == "__main__":
    # --- basic config for quick testing ---
    conf = Config(
        gamma=0.95,
        trembling_hand=0.02,
        skeleton_K=2,
        only_skeleton=True,
        num_total_strat=1000,  # ignored when only_skeleton=True
        mutation=0.0,
        dt=0.05,
        child_sigma=0.05,
        redistribution_rate=0.05,
        b_NS=2.5,  # b_NS: float = 2.0,  b_NS > 1
        c_NS=0.5,  # c_NS: float = 0.5,  c_NS > 0
    )
    thresh = 0.000000002
    replicator_steps = 1000
    num_iter = 50

    print("Building PopulationDyn...")
    popdyn = PopulationDyn(conf, device="cpu")

    print(f"N_tot = {popdyn.N_tot}, N_skel = {popdyn.N_skel}, N_mut = {popdyn.N_mut}, N_repl = {popdyn.N_repl}")
    print("Initial x sum:", popdyn.x.sum().item())

    # --- run quick test ---
    print("Running some dynamics")
    for i in range(num_iter):
        popdyn.run_replicator_dyn(steps=replicator_steps)
        # print("\nSurvivors after i =", i+1)
        # popdyn.pop.compute_VX_matrix()
        popdyn.print_survivors(threshold=thresh)
        print(i, popdyn.x @  popdyn.pop.p)
