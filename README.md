# PytorchDRL 

Pytorch-based Deep Reinforcement Learning framework.

The project follows a clear abstraction hierarchy for DRL. 
The goal is conceptual clarity, extensibility across algorithms (PPO, DQN, etc.), 
extensibility across network architectures (MLP, Conv1d, Conv2d, LSTM, etc.),
and long-term maintainability. 
The design is based on four core abstractions: 
Environment, Model (Network), Agent, and Trainer. 

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
  - Handle training logic

### 2. Model (The brain)
- **What it is:** A pure function approximator (`nn.Module`)
- **Responsibilities:** 
  - Map observations to predictions
  - Provide differentiable outputs
- **Does NOT:** 
  - Sample actions
  - Implement algorithm logic
- Each model is one of these variants:
    - Policy-value: `forward(obs) -> (policy_logits, value)`
    - Q-value: `forward(obs) -> q_values`
    - Policy-only: `forward(obs) -> policy_logits`

### 3. Agent (The student)
- **What it is:** The actor that interacts with the environment
- **Responsibilities:** 
  - Own model
  - Select action given observation (evaluation, deployment support)
  - Select action and return model outputs given observation (trainer rollout support)
- **Does NOT:** 
  - Own optimizers, 
  - Perform training logic
  - Know about the algorithm that trained it 
- Each agent uses one of the model variants. Agent variants are linked to model variants, 
NOT to trainer algorithm variants.
  - Policy-value model -> Policy-value agent
  - Q-value model -> Q-value agent
  - Policy-only model -> Policy-only agent
- Agents are intentionally thin and stable. They are designed to survive the training process.

### 4. Trainer (The teacher)
- **What it is:** The implementation of an algorithm that performs one single learning step.
- **Responsibilities:** 
  - Own environment, agent, optimizer
  - Own and manage algorithm-specific buffer
  - Collect experience (rollout)
  - Update agent's model
- **Does NOT:**
  - Directly own model (except target model with DQN)
  - Schedule multistep learning process
- The trainer's algorithm has to be compatible with its agent variant. Examples:
    - PPO -> Policy-value agent
    - DQN -> Q-value agent
- The trainer performs only one learning step, but may use experience from previous steps.

### Key Takeaway
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
│   │   └── evaluator.py          # Evaluator implemented  
│   │   
│   ├── models/               # Library of pre-built frozen Model implementations
│   │   ├── policy_value/         # forward(obs) -> (policy, value)
│   │   │   ├── mlp.py
│   │   │
│   │   ├── q_value/              # forward(obs) -> q_values
│   │   │   ├── mlp.py
│   │   │
│   │   └── policy_only/          # forward(obs) -> policy
│   │       ├── mlp.py 
│   │          
│   ├── agents/               # Frozen Agent implementations (based on Model variants)
│   │   ├── policy_value_agent.py
│   │   ├── q_value_agent.py
│   │   └── policy_only_agent.py
│   │
│   └── trainers/             # Library of pre-built frozen Trainer implementations
│       ├── ppo.py                # PPOTrainer + PPOBuffer
│       ├── dqn.py                # DQNTrainer + DQNBuffer
│
└── worlds/             # User extension space for concrete worlds
    ├── gridworld/          # Simple navigation problem
    │   ├── environment.py
    │   ├── model.py
    │   └── test01.py
    ├── ipd/                # Iterated Prisoner's Dilemma problem
    │   ├── environment.py
    │   ├── model.py
    │   └── test01.py
    ├── other/
```

Each level answers a different question:

- models/ - How does the brain compute?
- agents/ - How does an actor behave given a brain?
- trainers/ - How does learning improve the actor / change the brain?
- worlds/ - Formulation of the specific problem
  - What specific environment does the actor interact with? 
  - What specific model is used? 
  - What is the specific problem to be solved?

## How to Create a Custom World
Make a subdirectory under worlds/

### 1. Custom environment
Implement your own CustomEnvironment class, derived from the base Environment class. 
These interfaces MUST be implemented:
```python
from drl.common.interfaces import Environment
from drl.common.types import Observation, Action, Reward, Done

class CustomEnvironment(Environment):

    @property
    def batch_size(self) -> int: ...
       
    @property
    def random_termination(self) -> bool: ...
       
    @property
    def gamma(self) -> float: ...
        
    @property
    def obs_template(self) -> tuple[int, ...]: ...
        
    @property
    def num_actions(self) -> int: ...

    def reset_state(self, mask: Tensor) -> None: ...

    def apply(self, action: Action) -> tuple[Reward, Done]: ...

    def get_obs(self) -> Observation: ...
```

### 2. Custom model
Choose a model from the model library, or implement your own 
CustomModel class, derived from one of the base Model class variants. 
For example, you want a custom policy-value model for ppo training. 
The forward() interface MUST be implemented:
```python
from drl.common.types import Observation, PolicyLogits, Value
from drl.common.interfaces import PolicyValueModel

class CustomPolicyValueModel(PolicyValueModel):

    def forward(self, obs: Observation) -> tuple[PolicyLogits, Value]: ...
```
Make sure Observation type and PolicyLogits type in your model implementation 
are consistent with Observation type and Action type in your 
environment implementation. 

### 3. Agent, trainer, custom script
Agent variants and trainer variants are already implemented. Following up on
the ppo training example, your minimal script may look like this:
```python
from drl.agents.policy_value_agent import PolicyValueAgent
from drl.trainers.ppo import PPOTrainer
from drl.common.evaluator import Evaluator

# set up configuration parameters: batch_size, gamma, custom_parameters

env_train = CustomEnvironment(batch_size, gamma, custom_parameters, random_termination=True)
env_eval = CustomEnvironment(batch_size, gamma, custom_parameters, random_termination=False)
print("env_train and env_eval are ready")

model = CustomPolicyValueModel(custom_parameters)
print("model is ready")

agent = PolicyValueAgent(model=model, custom_parameters)
print("agent is ready")

trainer = PPOTrainer(env=env_train, agent=agent,
                     rollout_length=128, epochs=4, mini_batch=64,
                     lam=0.9, clip_eps=0.2, lr=0.0001)
print("trainer is ready")

evaluator = Evaluator(env=env_eval, agent=agent)
print("evaluator is ready")

# ... your training script ...
```
