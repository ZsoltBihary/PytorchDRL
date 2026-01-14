from torch.utils.data import DataLoader
from src.config import Config
from src.environment import Environment
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from line_profiler_pycharm import profile


class DQNTrainer:
    def __init__(self, conf: Config, train_env: Environment, eval_env: Environment,
                 agent: DQNAgent, buffer: ReplayBuffer):
        self.conf = conf
        # Extract config parameters
        self.rollout_steps = conf.rollout_steps
        self.num_epochs = conf.num_epochs
        self.minibatch_size = conf.minibatch_size
        self.target_update = conf.target_update
        self.lr0 = conf.learning_rate
        self.eps0 = conf.epsilon
        self.temp0 = conf.temperature
        # Set up necessary objects
        self.train_env = train_env
        self.eval_env = eval_env
        self.agent = agent
        self.buffer = buffer

    # ------------------------------------------------------------
    # Rollout (collect experience)
    # ------------------------------------------------------------
    @profile
    def rollout(self) -> float:
        self.agent.online_model_cpu.eval()
        sum_reward = 0.0
        state = self.train_env.get_state()

        for _ in range(self.rollout_steps):
            action = self.agent.act(state, greedy=False)
            next_state, reward = self.train_env.step(action)
            sum_reward += reward.mean().item()
            self.buffer.push(
                state,
                action,
                reward,
                next_state
            )
            state = next_state
        return sum_reward / self.rollout_steps

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    @profile
    def evaluate(self) -> float:
        self.agent.online_model_cpu.eval()
        sum_reward = 0.0
        state = self.eval_env.reset()
        eval_mult = 2
        for _ in range(eval_mult * self.rollout_steps):
            action = self.agent.act(state, greedy=True)
            next_state, reward = self.eval_env.step(action)
            sum_reward += reward.mean().item()
            state = next_state
        return sum_reward / self.rollout_steps / eval_mult

    # ------------------------------------------------------------
    # Training (optimize Q network)
    # ------------------------------------------------------------
    @profile
    def train(self):
        if len(self.buffer) == 0:
            return 0.0
        self.agent.online_model.train()
        self.agent.target_model.eval()

        dataloader = DataLoader(self.buffer, batch_size=self.minibatch_size, shuffle=True)

        losses = []
        for _ in range(self.num_epochs):
            for batch in dataloader:
                loss = self.agent.train_step(batch)
                losses.append(loss.item())

        return sum(losses) / len(losses)

    # ------------------------------------------------------------
    # One full cycle of rollout + training + evaluation
    # ------------------------------------------------------------
    @profile
    def cycle(self, cycle_idx):
        avg_ro_rew = self.rollout()
        avg_loss = self.train()
        # after each update → sync CPU rollout model
        self.agent.sync_cpu_model()
        avg_ev_rew = self.evaluate()

        if cycle_idx % self.target_update == 0:
            self.agent.update_target_model()

        return avg_loss, avg_ro_rew, avg_ev_rew

    # ------------------------------------------------------------
    # Train for N cycles
    # ------------------------------------------------------------
    @profile
    def run(self, n_cycles, verbose=True):
        for i in range(1, n_cycles + 1):

            # Update scheduled parameters
            progress = i / n_cycles
            self.agent.lr = max(self.lr0 / 100.0, self.lr0 * (1.0 - 1.5 * progress))
            self.agent.epsilon = max(self.eps0 / 5.0, self.eps0 * (1.0 - 1.2 * progress))
            self.agent.temperature = max(self.temp0 / 10.0, self.temp0 * (1.0 - progress))

            # One full cycle of rollout + training + evaluation
            avg_loss, avg_ro_rew, avg_ev_rew = self.cycle(i)
            if verbose:
                print(
                    f"[{i:03d}/{n_cycles}] "
                    f"buffer {len(self.buffer)}/{self.buffer.capacity} | "
                    f"lr= {self.agent.lr:.6f} | "
                    f"eps= {self.agent.epsilon:.3f} | "
                    f"temp= {self.agent.temperature:.4f} | "
                    f"avg_loss= {avg_loss:.4f} | "
                    f"avg_rollout_rew= {avg_ro_rew:.4f} | "
                    f"avg_eval_rew= {avg_ev_rew:.4f} "
                )


if __name__ == "__main__":
    conf = Config(
        batch_size=128,
        window_size=1,
        rollout_steps=100,
        buffer_mult=2,
        learning_rate=0.001,
        use_ddqn=True,
        use_layernorm=True,
        num_blocks=3,
        hidden_dim=32
    )

    train_env = Environment(conf)
    eval_env = Environment(conf)
    agent = DQNAgent(conf)
    buffer = ReplayBuffer(conf)

    trainer = DQNTrainer(conf, train_env, eval_env, agent, buffer)

    trainer.run(n_cycles=50)
    print("Training finished.")
