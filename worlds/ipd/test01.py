# worlds/ipd/test01.py

import torch
from torchinfo import summary
from drl.agents.policy_value_agent import PolicyValueAgent
from drl.trainers.ppo import PPOTrainer
from drl.common.evaluator import Evaluator
from worlds.ipd.prepare_pool import prepare_pool
from worlds.ipd.environment import IteratedPrisonersDilemma
from worlds.ipd.models.conv1d import PolicyValueConv1D

p_pool, w_pool, gamma, payoffs = prepare_pool()
B = 128
L = 20
env = IteratedPrisonersDilemma(batch_size=B, gamma=gamma, random_termination=True, history_len=L,
                               opponent_probs=p_pool, opponent_weights=w_pool, payoffs=payoffs)
env_eval = IteratedPrisonersDilemma(batch_size=B, gamma=gamma, random_termination=False, history_len=L,
                                    opponent_probs=p_pool, opponent_weights=w_pool, payoffs=payoffs)

print("env and env_val are ready")
model = PolicyValueConv1D(L=L, obs_channels=env.obs_template.shape[1], action_dim=2,
                          trunk_channels=32, num_res_blocks=2,
                          policy_hidden=16, value_hidden=4)
dummy_obs = torch.randn(B, L, 5)
# Print model summary
summary(model, input_data=dummy_obs,
        col_names=["input_size", "output_size", "num_params"],
        depth=2, verbose=1)
print("model is ready")
agent = PolicyValueAgent(model=model, deterministic=False)
print("agent is ready")
trainer = PPOTrainer(env=env, agent=agent,
                     rollout_length=128, epochs=2, mini_batch=128,
                     lam=0.95, clip_eps=0.2, lr=0.002)
print("trainer is ready")
evaluator = Evaluator(env=env_eval, agent=agent)
print("evaluator is ready")
returns = evaluator.run()
print("mean_return:", (1.0 - gamma) * returns.mean())

n_train = 100
obs = env_eval.reset()
for i in range(n_train):
    result = trainer.step()
    print(i)
    print(result)
    returns = evaluator.run()
    print("mean_return:", (1.0 - gamma) * returns.mean())
