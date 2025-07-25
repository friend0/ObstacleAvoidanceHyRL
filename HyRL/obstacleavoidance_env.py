import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io


@dataclass
class State:
    x: float
    y: float

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y)[index]

    def __len__(self) -> int:
        return 2


@dataclass
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Obstacle:
    center: Point
    r: float


class ObstacleAvoidance(gym.Env):
    def __init__(
        self,
        steps=120,
        bounds: BBox = BBox(x_min=0.0, x_max=3.0, y_min=-1.5, y_max=1.5),
        obstacle: Obstacle = Obstacle(center=Point(x=1.5, y=0.0), r=0.75),
        goal: Point = Point(x=3.0, y=0.0),
        t_sampling=0.05,
        random_init=True,
        state_init=np.array([0.0, 0.0], dtype=np.float32),
        spread=np.array([0.5, 1.0], dtype=np.float32),
        backwards=False,
        hybridlearning=False,
        M_ext=None,
        render_mode=None,
    ):
        self.steps = steps
        self.random_init = random_init
        self.state_init = state_init
        self.spread = spread
        self.backwards = backwards
        self.hybridlearning = hybridlearning
        self.M_ext = M_ext
        self.bounds = bounds
        self.obstacle = obstacle
        self.goal = goal
        self.t_sampling = t_sampling

        self.x_obst, self.y_obst = obstacle.center.x, obstacle.center.y
        self.radius_obst = obstacle.r
        self.x_goal, self.y_goal = goal.x, goal.y

        self.low_state = np.array([0.0, 0.0, bounds.y_min], dtype=np.float32)
        self.high_state = np.array([3, 4.5, bounds.y_max], dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        self._action_to_direction = {
            0: np.array([1, 1]),  # left
            1: np.array([1, 0.5]),  # slight left
            2: np.array([1, 0]),  # stright
            3: np.array([1, -0.5]),  # slight right
            4: np.array([1, -1]),  # right
        }
        self.np_random = None
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_observation(self):
        dist_obst = max(
            np.sqrt((self.x - self.x_obst) ** 2 + (self.y - self.y_obst) ** 2)
            - self.radius_obst,
            0.0,
        )
        dist_goal = np.sqrt((self.x - self.x_goal) ** 2 + (self.y - self.y_goal) ** 2)
        self.state = np.array([dist_obst, dist_goal, self.y], dtype=np.float32)

    def check_terminate(self):
        self.terminate = (
            self.state[0] <= 65e-3
            or abs(self.y) >= self.bounds.y_max
            or self.x > self.bounds.x_max
        )

    def step(self, action):
        dx, dy = self._action_to_direction[int(action)]
        sign = 1 if not self.backwards else -1

        if self.hybridlearning:
            if not self.M_ext.in_M_ext(np.array([self.x, self.y], dtype=np.float32)):
                sign = 0

        # Apply motion
        self.x += sign * dx * self.t_sampling
        self.y += sign * dy * self.t_sampling

        # Update state
        self.update_observation()

        # --- Reward Components ---
        x_error = abs(self.x_goal - self.x)
        y_error = abs(self.y_goal - self.y)
        dist_to_goal = np.sqrt(x_error**2 + y_error**2)

        # Normalize distance to goal
        max_possible_dist = np.linalg.norm([self.bounds.x_max, self.bounds.y_max])
        goal_progress = 1 - (dist_to_goal / max_possible_dist)  # range [0, 1]

        # Obstacle penalty (barrier function)
        dist_to_obstacle = self.state[0]
        barrier = ((dist_to_obstacle - self.radius_obst) / self.radius_obst) ** 2
        obstacle_penalty = -0.05 * barrier  # softer repulsion

        # Step penalty: encourage shorter paths
        step_penalty = -0.005

        # Combine reward terms
        raw_reward = goal_progress + obstacle_penalty + step_penalty

        # Smooth the reward
        reward = 5 * np.tanh(raw_reward)

        # Completion bonus
        if dist_to_goal < 0.1:
            reward += 10
            self.terminate = True

        # Final bookkeeping
        self.steps_left -= 1
        self.check_terminate()
        terminated = self.terminate
        truncated = self.steps_left <= 0
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.terminate = False
        self.x, self.y = self.state_init[0], self.state_init[1]

        if self.random_init:
            self.x = np.float32(
                self.state_init[0] + self.spread[0] * self.np_random.uniform(0, 1)
            )
            self.y = np.float32(
                self.state_init[1] + self.spread[1] * self.np_random.uniform(-1.0, 1.0)
            )

            if self.hybridlearning:
                while not self.M_ext.in_M_ext(
                    np.array([self.x, self.y], dtype=np.float32)
                ):
                    self.x = np.float32(
                        self.state_init[0]
                        + self.spread[0] * self.np_random.uniform(0, 1)
                    )
                    self.y = np.float32(
                        self.state_init[1]
                        + self.spread[1] * self.np_random.uniform(-1.0, 1.0)
                    )

        self.update_observation()
        self.steps_left = self.steps

        return (
            self.state.copy(),
            {},
        )  # 👈 .copy() avoids Gym warnings about shared memory

    def render(self, mode="rgb_array"):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

        # Plot bounds
        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)

        # Obstacle
        obstacle = plt.Circle(
            (self.x_obst, self.y_obst), self.radius_obst, color="gray"
        )
        ax.add_patch(obstacle)

        # Goal
        ax.plot(self.x_goal, self.y_goal, "go", markersize=12, label="Goal")

        # Agent
        ax.plot(self.x, self.y, "bo", markersize=8, label="Agent")

        # Formatting
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        # Render to RGB array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf, dtype=np.uint8)[..., :3]  # Strip alpha
        plt.close(fig)
        return image
