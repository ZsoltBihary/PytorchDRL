import torch
from torch import Tensor

# ===== Aliases to make type-setting more expressive =====
PriceSeq = Tensor       # shape: (B, T)
Position = Tensor       # shape: (B, )
State = tuple[PriceSeq, Position]
QValues = Tensor        # shape: (B, A)
Action = Tensor         # shape: (B,)
Reward = Tensor         # shape: (B,)


def action_2_pos(action: Tensor) -> Tensor:
    """ Convert action [0, 1, 2] to position [-1.0, 0.0, 1.0] """
    return (action - 1).to(dtype=torch.float32)


def pos_2_action(pos: Tensor) -> Tensor:
    """ Convert position [-1.0, 0.0, 1.0] to action [0, 1, 2] """
    return pos.to(dtype=torch.long) + 1


class Config:

    def __init__(
            self,
            # ===== Data tensor sizes =====
            num_actions: int = 3,            # A, number of possible actions
            batch_size: int = 128,           # B, number of parallel simulations
            window_size: int = 1,            # T, lookback window size

            # ===== Price dynamics =====
            S_mean: float = 100.0,          # mean-reversion price level
            volatility: float = 2.0,        # daily volatility
            mean_reversion: float = 0.1,    # inverse mean-reversion timescale

            # ===== Reward specification =====
            gamma: float = 0.99,            # one-period discount factor used in PV(reward)
            half_bidask: float = 1.0,       # bid-ask trading friction parameter
            risk_aversion: float = 0.02,    # weight on variance in mean-variance utility

            # ===== QModel specification =====
            num_blocks=3,                   # number of residual blocks
            hidden_dim: int = 32,           # number of hidden features in Q-model
            use_layernorm=False,            # flag for layer norm
            dropout=0.00,                   # dropout = 0.0 means no dropout
            dueling=True,                   # flag for dueling heads

            # ===== Exploration parameters =====
            epsilon: float = 0.25,          # initial epsilon for epsilon-greedy action selection
            temperature: float = 0.001,     # initial temperature for soft action selection

            # ===== Training cycle control =====
            train_on_CUDA: bool = True,
            rollout_steps: int = 100,       # R, number of rollout steps
            buffer_mult: int = 2,           # number of rollouts that fills the replay buffer
            learning_rate: float = 0.0050,  # initial learning rate for optimization
            use_ddqn: bool = True,          # flag for double deep Q-learning
            num_epochs: int = 1,            # number of epochs in one training cycle
            minibatch_size: int = 256,      # minibatch size used in one training step
            target_update: int = 1          # number of rollout+training cycles between target network updates
    ):
        # ===== Data tensor sizes
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.window_size = window_size
        # ===== Price dynamics
        self.S_mean = S_mean
        self.volatility = volatility
        self.mean_reversion = mean_reversion
        # ===== Reward specification
        self.gamma = gamma
        self.half_bidask = half_bidask
        self.risk_aversion = risk_aversion
        # ===== QModel specification
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm
        self.dropout = dropout
        self.dueling = dueling
        # ===== Exploration parameters
        self.epsilon = epsilon
        self.temperature = temperature
        # ===== Training cycle control
        self.train_on_CUDA = train_on_CUDA
        self.rollout_steps = rollout_steps
        self.buffer_mult = buffer_mult
        self.buffer_capacity = buffer_mult * rollout_steps * batch_size
        self.learning_rate = learning_rate
        self.use_ddqn = use_ddqn
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.target_update = target_update
