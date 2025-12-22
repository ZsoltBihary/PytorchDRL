# envs/gridworld/environment.py

import torch
from drl.common.interfaces import Environment
from drl.common.types import Observation, Action, Reward, Done


class GridWorld(Environment):

    def __init__(self, batch_size: int, gamma: float, random_termination: bool,
                 height: int = 5, width: int = 5):
        # consume parameters
        super().__init__()
        self.B = batch_size
        self._gamma = gamma
        self._rand_term = random_termination
        self._obs_shape = (2,)
        self._num_actions = 4
        self.H = height
        self.W = width

        # internal state: agent position (B, 2)
        self.agent_pos = torch.zeros((self.B, 2), dtype=torch.long)
        # constant goal position at center (2, )
        self.goal_pos = torch.tensor(data=[self.H // 2, self.W // 2], dtype=torch.long)
        # constant reset position at a corner (2, )
        self.reset_pos = torch.zeros(2, dtype=torch.long)
        # four teleport positions near corners (4, 2)
        self.teleport_pos = torch.tensor(data=[
                [1, 1],
                [1, self.W - 2],
                [self.H - 2, 1],
                [self.H - 2, self.W - 2]
        ], dtype=torch.long)
        # movement deltas indexed by discrete actions 0..3
        self.delta = torch.tensor(data=[
                [1, 0],    # down
                [0, 1],    # right
                [-1, 0],   # up
                [0, -1]    # left
        ], dtype=torch.long)
        # normalizer for observation
        self.norm = torch.tensor(data=[self.H - 1, self.W - 1], dtype=torch.float32)

    # ---------------------------------------------------
    # required properties for Environment interface
    # ---------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.B

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self._obs_shape

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def random_termination(self) -> bool:
        return self._rand_term

    # ---------------------------------------------------
    # required methods for Environment interface
    # ---------------------------------------------------
    def reset_state(self, mask: torch.Tensor) -> None:
        """
        mask: BoolTensor (B,)
        """
        self.agent_pos[mask] = self.reset_pos

    def apply(self, action: Action) -> tuple[Reward, Done]:
        """
        Apply environment dynamics for all envs.

        Args:
            action: LongTensor shape (B,)
        Returns:
            reward: FloatTensor (B,)
            done: FloatTensor binary mask shape (B,)
        """
        # integer delta movement (B,2)
        move = self.delta[action]
        # apply movement, clamped to boundaries
        self.agent_pos += move
        self.agent_pos[:, 0].clamp_(min=0, max=self.H - 1)
        self.agent_pos[:, 1].clamp_(min=0, max=self.W - 1)
        # default reward
        reward = -torch.ones(self.B, dtype=torch.float32)
        # detect goal reached, and teleport
        reached_goal = torch.all(self.agent_pos == self.goal_pos, dim=-1)
        if reached_goal.any():
            reward[reached_goal] += 10.0  # task reward
            self.teleport(reached_goal)  # continuing teleport

        # continuing env: always done=0.0
        done = torch.zeros(self.B, dtype=torch.float32)
        return reward, done

    def get_obs(self) -> Observation:
        """
        Normalized float positions in [0,1]
        shape: (B,2)
        """
        return self.agent_pos.float() / self.norm

    def teleport(self, mask: torch.Tensor) -> None:
        n = mask.sum().item()
        # sample teleport positions for each selected
        idx = torch.randint(low=0, high=len(self.teleport_pos), size=(n,))
        self.agent_pos[mask] = self.teleport_pos[idx]

    # ---------------------------------------------------
    # visualization
    # ---------------------------------------------------
    def render(self):
        """
        Print only the first env in batch.
        """
        grid = [["." for _ in range(self.W)] for _ in range(self.H)]

        ax, ay = self.agent_pos[0].tolist()
        gx, gy = self.goal_pos.tolist()

        grid[gx][gy] = "G"
        grid[ax][ay] = "A"

        for row in grid:
            print(" ".join(row))
        print()


if __name__ == "__main__":
    env = GridWorld(batch_size=3, height=5, width=5, gamma=0.9, random_termination=True)

    print("\n=== Reset ===")
    obs = env.reset()
    # print("obs:", obs)
    env.render()

    print("=== Random rollout ===")
    for t in range(60):
        # sample random discrete actions (0..3)
        act = torch.randint(low=0, high=4, size=(env.batch_size,))
        obs, rew, done = env.step(act)

        print("action:", act[0].item(), "reward:", rew[0].item(), "done:", done[0].int().item())
        env.render()
