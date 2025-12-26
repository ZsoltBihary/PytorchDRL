# envs/gridworld/test01.py

from worlds.gridworld.environment import GridWorld
from drl.models.policy_value.mlp import PolicyValueMLP
from drl.agents.policy_value_agent import PolicyValueAgent
from drl.trainers.ppo import PPOTrainer
from drl.common.evaluator import Evaluator

H, W, gamma = 7, 7, 0.9
env = GridWorld(batch_size=256, gamma=gamma, random_termination=True, height=H, width=W)
env_eval = GridWorld(batch_size=256, gamma=gamma, random_termination=False, height=H, width=W)
print("env and env_val are ready")
obs_dim = env.obs_shape[0]
num_actions = env.num_actions
model = PolicyValueMLP(obs_dim=obs_dim, action_dim=num_actions, hidden_dim=32)
print("model is ready")
agent = PolicyValueAgent(model=model)
print("agent is ready")
trainer = PPOTrainer(env=env, agent=agent,
                     rollout_length=64, epochs=5, mini_batch=64,
                     lam=0.9, clip_eps=0.2, lr=0.001)
print("trainer is ready")
evaluator = Evaluator(env=env_eval, agent=agent, max_steps=100)
print("evaluator is ready")

# obs = env_eval.reset()
# env_eval.render()
for i in range(10):
    result = trainer.step()
    print(i)
    print(result)
    returns = evaluator.run()
    print("mean_return:", (1.0 - gamma) * returns.mean())

    # print("=== Test strategy")
    # for t in range(10):
    #     # sample random discrete actions (0..3)
    #     action = agent.act(obs)
    #     obs, rew, done = env_eval.step(action)
    #
    #     print("action:", action[0].item(), "reward:", rew[0].item(), "done:", done[0].int().item())
    #     env_eval.render()

a = 42
