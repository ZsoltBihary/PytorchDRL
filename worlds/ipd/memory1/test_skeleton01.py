# import torch
from config import Config
# from strat_population import Memory1Population
from population_dyn import PopulationDyn

if __name__ == "__main__":
    conf = Config()
    thresh = 0.0001
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

    p_pool, w_pool = popdyn.extract_pool()
    a = 42
