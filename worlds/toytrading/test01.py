# worlds/toytrading/test01.py

# import torch
# import torch.nn as nn
# from torchinfo import summary
from drl.common.torchinfo_tree_summary import torchinfo_tree_summary
from drl.agents.policy_value_agent import PolicyValueAgent
from drl.trainers.ppo import PPOTrainer
from drl.common.evaluator import Evaluator
import drl.common.tensor_tree as tt
from worlds.toytrading.config import Config
from worlds.toytrading.environment import ToyTrading
from worlds.toytrading.model import ToyTradingModelMLP

conf = Config()
B = conf.batch_size
gamma = conf.gamma

env = ToyTrading(batch_size=B, gamma=gamma, random_termination=True, conf=conf)
env_eval = ToyTrading(batch_size=B, gamma=gamma, random_termination=False, conf=conf)
print("env and env_val are ready")

print("\n=== env OBS_TEMPLATE SUMMARY ===")
tt.summary_tree(env.obs_template)

model = ToyTradingModelMLP(obs_template=env.obs_template, action_dim=env.num_actions, hidden_dim=32)
dummy_obs = env.get_obs()
# Print model summary
obs = env.get_obs()
torchinfo_tree_summary(model, obs, col_names=["input_size", "output_size", "num_params"], depth=3)
print("model is ready")

agent = PolicyValueAgent(model=model, deterministic=False)
print("agent is ready")
trainer = PPOTrainer(env=env, agent=agent,
                     rollout_length=512, epochs=2, mini_batch=128,
                     lam=0.95, clip_eps=0.2, lr=0.001)
print("trainer is ready")
evaluator = Evaluator(env=env_eval, agent=agent)
print("evaluator is ready")

returns = evaluator.run()
print("mean_return:", (1.0 - gamma) * returns.mean())
obs = env_eval.reset()

for i in range(50):
    result = trainer.step()
    print(i)
    print(result)
    returns = evaluator.run()
    print("mean_return:", (1.0 - gamma) * returns.mean())
