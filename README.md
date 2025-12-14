# PytorchDRL
Research-oriented PyTorch implementations of deep reinforcement learning algorithms (PPO, DQN) with clean interfaces.
## Project Structure
PytorchDRL/
├── README.md
├── pyproject.toml              # optional
│
├── algorithms/                 # algorithm-level logic (generic)
│   ├── __init__.py
│   │
│   ├── common/                 # shared algorithm abstractions
│   │   ├── __init__.py
│   │   └── env_interface.py    # EnvBatch ABC
│   │
│   ├── ppo/
│   │   ├── __init__.py
│   │   ├── interfaces.py       # PPO-specific ABCs
│   │   ├── buffer.py           # RolloutBuffer
│   │   ├── trainer.py          # PPOTrainer
│   │   └── networks/
│   │       ├── __init__.py
│   │       ├── base.py         # ActorCriticNet ABC
│   │       ├── mlp.py
│   │       ├── conv1d.py
│   │       └── resconv1d.py
│   │
│   └── dqn/                    # future
│       ├── __init__.py
│       ├── interfaces.py
│       ├── buffer.py
│       ├── trainer.py
│       └── networks/
│           ├── __init__.py
│           └── qnet.py
│
├── environments/               # problem domains
│   └── ipd/
│       ├── __init__.py
│       ├── memory1/            # existing code
│       │   ├── strategies.py
│       │   └── population.py
│       ├── env.py              # IPDEnv
│       ├── observations.py
│       └── config.py
│
├── experiments/                # glue code
│   ├── ipd_ppo_train.py
│   ├── ipd_ppo_sweep.py
│   └── debug_env.py
│
├── common/                     # small shared helpers
│   ├── typing.py
│   └── torch_utils.py
│
└── tests/
    ├── test_rollout_buffer.py
    ├── test_ipd_env.py
    └── test_ppo_trainer.py
