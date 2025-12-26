# worlds/ipd/test01.py

from worlds.ipd.prepare_pool import prepare_pool
from worlds.ipd.environment import IteratedPrisonersDilemma

p_pool, w_pool, gamma = prepare_pool()
B = 32
L = 10
env = IteratedPrisonersDilemma(batch_size=B, history_len=L,
                               opponent_probs=p_pool, opponent_weights=w_pool,
                               gamma=gamma)
