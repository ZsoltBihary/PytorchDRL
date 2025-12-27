# worlds/ipd/prepare_pool.py


def prepare_pool():

    from worlds.ipd.memory1.config import Config
    from worlds.ipd.memory1.population_dyn import PopulationDyn

    conf = Config()
    thresh = 0.0001
    replicator_steps = 1000
    num_iter = 50

    print("Building PopulationDyn...")
    popdyn = PopulationDyn(conf, device="cpu")
    print(f"N_tot = {popdyn.N_tot}, N_skel = {popdyn.N_skel}, N_mut = {popdyn.N_mut}, N_repl = {popdyn.N_repl}")

    print("Running replicator dynamics ...")
    for i in range(num_iter):
        popdyn.run_replicator_dyn(steps=replicator_steps)

    popdyn.print_survivors(threshold=thresh)
    p_pool, w_pool = popdyn.extract_pool(threshold=thresh)
    return p_pool, w_pool, conf.gamma, conf.payoffs


if __name__ == "__main__":
    p_pool, w_pool, gamma, payoffs = prepare_pool()
    print("p_pool =", p_pool)
    print("w_pool =", w_pool)
    print("gamma =", gamma)
    print("payoffs =", payoffs)
