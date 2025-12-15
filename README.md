# PytorchDRL – Conceptual Architecture

This project follows a clear, algorithm-agnostic abstraction hierarchy for deep reinforcement learning. The goal is conceptual clarity, extensibility across algorithms (PPO, DQN, etc.), and long-term maintainability. The design is based on four core abstractions: Environment, Model (Network), Agent, and Trainer. These abstractions cleanly separate what the world is, what the model computes, what the algorithm is, and how learning is orchestrated.

## Core Abstractions

### 1. Environment
- **What it is:** A simulator of environment dynamics.
- **Responsibilities:** Store environment state; apply actions; produce observations, rewards, and termination flags.
- **Does NOT:** Learn, know about neural networks, or handle training logic.
- **Typical interface:**
    obs = env.reset()
    obs, reward, done = env.step(action)

### 2. Model (Network)
- **What it is:** A pure function approximator (always an `nn.Module`).
- **Responsibilities:** Map observations to predictions; provide differentiable outputs.
- **Examples:**
    - PPO: `forward(obs) -> (policy_logits, value)`
    - DQN: `forward(obs) -> q_values`
    - Policy-only: `forward(obs) -> policy_logits`
- **Does NOT:** Sample actions, store experience, or implement algorithm logic.

### 3. Agent (the algorithm)
- **What it is:** The intelligence of the system; algorithm-specific logic.
- **Responsibilities:** Own model(s); define how actions are selected; define how actions are evaluated for training; implement algorithm-specific computations.
- **Typical methods:**
    action = agent.act(obs)
    metrics = agent.evaluate_actions(obs, actions)
    loss = agent.compute_loss(batch)
- **Does NOT:** Own optimizers, control training loops, or manage rollout length or epochs.
- The Agent is what remains usable after training, during evaluation or deployment.

### 4. Trainer (the learning process)
- **What it is:** Orchestrator of learning.
- **Responsibilities:** Own environment(s), buffer(s), and optimizer(s); run rollout collection; run training loops; call Agent methods.
- **Does NOT:** Implement algorithm math, know model internals, or decide how actions are computed internally.

## Ownership Structure
```text
Trainer  
 ├── Environment  
 ├── Agent  
 │    ├── Model(s)  
 │    └── Policy logic  
 ├── Buffer  
 └── Optimizer
```

## Interaction Flow (Algorithm-Agnostic)

### Training Loop (High-Level)
1. Trainer resets environment  
2. Trainer asks Agent to act  
3. Environment steps  
4. Trainer stores transitions  
5. Rollout or replay buffer fills  
6. Trainer asks Agent to compute training quantities  
7. Trainer applies optimizer updates

### Example: PPO Interaction
    obs = env.reset()
    for t in rollout:
        action, logp, value = agent.act(obs)
        next_obs, reward, done = env.step(action)
        buffer.add(obs, action, reward, done, value, logp)
        obs = next_obs
    buffer.compute_gae(last_value=agent.estimate_value(obs))
    for batch in buffer:
        loss = agent.compute_loss(batch)
        optimizer.step()

### Example: DQN Interaction
    obs = env.reset()
    for t in steps:
        action = agent.act(obs)
        next_obs, reward, done = env.step(action)
        replay_buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if ready_to_update:
            batch = replay_buffer.sample()
            loss = agent.compute_loss(batch)
            optimizer.step()

## Evaluation and Deployment
- When training ends, the **Trainer disappears**; the **Agent remains**.
- Evaluation loop:
    obs = env.reset()
    while not done:
        action = agent.act(obs, mode="greedy")
        obs, reward, done = env.step(action)

## Key Takeaway
- **Environment** defines the world  
- **Model** approximates functions  
- **Agent** defines intelligence  
- **Trainer** defines learning  

This separation applies cleanly to PPO, DQN, and most modern DRL algorithms.

## Project Structure


```text
PytorchDRL/
├── README.md
│
├── algorithms/                 # algorithm-level logic (generic)
│   │
│   ├── common/                 # shared algorithm abstractions
│   │   └── env_interface.py    # Environment ABC
│   │
│   ├── ppo/
│   │   ├── interfaces.py       # PPO-specific ABCs
│   │   ├── buffer.py           # RolloutBuffer
│   │   ├── trainer.py          # PPOTrainer
│   │   └── networks/
│   │       ├── base.py         # ActorCritic ABC
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
