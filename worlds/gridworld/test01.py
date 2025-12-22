# envs/gridworld/test01.py

# import torch
import torch.nn as nn
from drl.common.interfaces import PolicyValueModel
from drl.agents.policy_value_agent import PolicyValueAgent
from drl.common.types import Observation, PolicyLogits, Value
from drl.trainers.ppo import PPOTrainer

from worlds.gridworld.environment import GridWorld


class PolicyValueMLP(PolicyValueModel):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # shared body for feature extraction
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )
        # policy head
        self.policy_head = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        # value head
        self.value_head = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]:
        x = self.net(obs)
        logits = self.policy_head(x)            # shape: (B, A)
        value = self.value_head(x).squeeze(-1)  # shape: (B,)
        return logits, value


env = GridWorld(batch_size=256, gamma=0.9, random_termination=True, height=5, width=5)
env_eval = GridWorld(batch_size=256, gamma=0.9, random_termination=True, height=5, width=5)
obs_dim = env.obs_shape[0]
num_actions = env.num_actions
model = PolicyValueMLP(obs_dim=obs_dim, action_dim=num_actions, hidden_dim=32)

obs = env.reset()
print("obs =", obs)
logits, value = model(obs)
print("logits =", logits)
print("value =", value)

agent = PolicyValueAgent(model=model)
print("agent is ready")
trainer = PPOTrainer(env=env, agent=agent,
                     rollout_length=128, epochs=6, mini_batch=64,
                     lam=0.9,
                     clip_eps=0.2, lr=0.0001)
print("trainer is ready")


obs = env_eval.reset()
env_eval.render()
for i in range(20):
    result = trainer.step()
    print(i)
    print(result)

    print("=== Test strategy")
    for t in range(10):
        # sample random discrete actions (0..3)
        action = agent.act(obs)
        obs, rew, done = env_eval.step(action)

        print("action:", action[0].item(), "reward:", rew[0].item(), "done:", done[0].int().item())
        env_eval.render()

a = 42
