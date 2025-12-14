# PytorchDRL
Research-oriented PyTorch implementations of deep reinforcement learning algorithms (PPO, DQN) with clean interfaces.

## Project Structure

```text
PytorchDRL/
├── README.md
│
├── algorithms/                 # algorithm-level logic (generic)
│   │
│   ├── common/                 # shared algorithm abstractions
│   │   └── env_interface.py    # EnvBatch ABC
│   │
│   ├── ppo/
│   │   ├── interfaces.py       # PPO-specific ABCs
│   │   ├── buffer.py           # RolloutBuffer
│   │   ├── trainer.py          # PPOTrainer
│   │   └── networks/
│   │       ├── base.py         # ActorCriticNet ABC
│   │       ├── mlp.py
│   │       ├── conv1d.py
│   │       └── resconv1d.py
│   │
│   └── dqn/                    # future
│       ├── interfaces.py
│       ├── buffer.py
│       ├── trainer.py
│       └── networks/
│           └── qnet.py
│
├── environments/               # problem domains
│   └── ipd/
│       ├── memory1/            # existing code
│       │   ├── strategies.py
│       │   └── population.py
│       ├── env.py              # IPDEnv
│       ├── observations.py
│       └── config.py
│
├── common/                     # small shared helpers
│   ├── typing.py
│   └── torch_utils.py
│
├── experiments/                # glue code
│   ├── ipd_ppo_train.py
│   ├── ipd_ppo_sweep.py
│   └── debug_env.py
│
└── tests/
    ├── test_rollout_buffer.py
    ├── test_ipd_env.py
    └── test_ppo_trainer.py
