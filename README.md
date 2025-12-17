# PytorchDRL 

## Conceptual Architecture

This project follows a clear abstraction hierarchy for deep reinforcement learning. 
The goal is conceptual clarity, extensibility across algorithms (PPO, DQN, etc.), and long-term maintainability. 
The design is based on four core abstractions: Environment, Model (Network), Agent, and Trainer. 

## Core Abstractions

### 1. Environment (The world)
- **What it is:** A simulator of environment dynamics
- **Responsibilities:** 
  - Maintain environment state 
  - Apply actions
  - Produce observations, rewards, and termination flags
- **Does NOT:** 
  - Learn
  - Know about neural networks
  - Handle training logic.

### 2. Model (The brain)
- **What it is:** A pure function approximator (`nn.Module`)
- **Responsibilities:** 
  - Map observations to predictions
  - Provide differentiable outputs
- **Does NOT:** 
  - Sample actions
  - Implement algorithm logic
- Each Model is one of these variants:
    - Policy-value: `forward(obs) -> (policy_logits, value)`
    - Q-value: `forward(obs) -> q_values`
    - Policy-only: `forward(obs) -> policy_logits`

### 3. Agent (The student)
- **What it is:** The trained actor that interacts with the environment
- **Responsibilities:** 
  - Own trained model(s)
  - Select action given observation
  - Support evaluation and deployment
- **Does NOT:** 
  - Own optimizers, 
  - Support training logic
  - Know about the algorithm that trained it
- Each Agent uses one of the Model variants
- Agents are intentionally thin and stable. They are designed to survive the training process.

### 4. Trainer (The teacher)
- **What it is:** The implementation of an algorithm that performs one single learning step.
- **Responsibilities:** 
  - Own environment, agent, optimizer
  - Own and manage algorithm-specific buffer
  - Run rollout collection, training loops
  - Compute algorithm-specific auxiliary variables using agent's Model
- **Does NOT:** Schedule multistep learning process
- The Trainer's algorithm has to be compatible with the Agent's Model variant. Examples:
    - PPO -> Policy-value
    - DQN -> Q-value
- The trainer performs only one learning step, but may use experience from previous steps.

## Key Takeaway
- **Environment** defines the world  
- **Model** approximates functions  
- **Agent** is a trainable actor  
- **Trainer** defines learning  

This separation applies cleanly to PPO, DQN, and most modern DRL algorithms.

## Project Structure

```text
PytorchDRL/
├── drl/                  # Future package root
│   ├── common/ 
│   │   ├── types.py              # Type aliases 
│   │   ├── interfaces.py         # Interfaces for Environment, Agent, and Model variants
│   │   └── utils.py              # Shared helpers  
│   │   
│   ├── models/               # Library of pre-built frozen Model implementations
│   │   ├── policy_value/         # forward(obs) -> (policy, value)
│   │   │   ├── mlp.py
│   │   │   └── other.py   
│   │   ├── q_value/              # forward(obs) -> q_values
│   │   │   ├── mlp.py
│   │   │   └── other.py   
│   │   └── policy_only/          # forward(obs) -> policy
│   │       ├── mlp.py 
│   │       └── other.py           
│   │   
│   ├── agents/               # Frozen Agent implementations (based on Model variants)
│   │   ├── policy_value_agent.py
│   │   ├── q_value_agent.py
│   │   └── policy_only_agent.py
│   │
│   └── trainers/             # Library of pre-built frozen Trainer implementations
│       ├── ppo.py                # PPOTrainer + PPOBuffer
│       ├── dqn.py                # DQNTrainer + DQNBuffer
│       └── other.py             
│
├── envs/               # User extension space for concrete environments
│   ├── ipd/                # Iterated Prisoner's Dilemma problem
│   │   ├── config.py
│   │   ├── environment.py
│   │   └── utilities.py
│   └── other/
│
└── experiments/
    ├── train_ppo_ipd.py
    └── other.py
```

Each level answers a different question:

- models/ - How does the brain compute?
- agents/ - How does an actor behave given a brain?
- trainers/ - How does learning improve the actor / change the brain?
- envs/ - What world does the actor interact with?
- experiments/ - What learning schedule is being run?
