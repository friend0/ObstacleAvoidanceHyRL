import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from dataclasses import dataclass


@dataclass
class State:
    x: float
    y: float

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("State supports indices 0 and 1 only.")

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
        steps=60,
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

        corners = np.array(
            [
                [bounds.x_min, bounds.y_min],
                [bounds.x_min, bounds.y_max],
                [bounds.x_max, bounds.y_min],
                [bounds.x_max, bounds.y_max],
            ],
            dtype=np.float32,
        )
        max_obst_dist = (
            np.max(
                np.linalg.norm(
                    corners - np.array([self.x_obst, self.y_obst], dtype=np.float32),
                    axis=1,
                )
            )
            - self.radius_obst
        )
        max_goal_dist = np.max(
            np.linalg.norm(
                corners - np.array([self.x_goal, self.y_goal], dtype=np.float32), axis=1
            )
        )

        self.low_state = np.array([0.0, 0.0, bounds.y_min], dtype=np.float32)
        self.high_state = np.array(
            [max_obst_dist, max_goal_dist, bounds.y_max], dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, shape=(3,), dtype=np.float32
        )
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_observation(self):
        # Compute the distance to the obstacle (clamped to zero if negative)
        dist_obst = max(
            np.sqrt((self.x - self.x_obst) ** 2 + (self.y - self.y_obst) ** 2)
            - self.radius_obst,
            0.0,
        )
        # Compute the distance to the goal.
        dist_goal = np.sqrt((self.x - self.x_goal) ** 2 + (self.y - self.y_goal) ** 2)
        self.state = np.array([dist_obst, dist_goal, self.y], dtype=np.float32)

    def check_terminate(self):
        self.terminate = (
            self.state[0] <= 1e-3
            or abs(self.y) >= self.bounds.y_max
            or self.x > self.bounds.x_max
        )

    def step(self, action):
        force = (action - 2) / 2
        sign = 1 if not self.backwards else -1

        if self.hybridlearning:
            if not self.M_ext.in_M_ext(np.array([self.x, self.y], dtype=np.float32)):
                sign = 0

        self.x += sign * self.t_sampling
        self.y += sign * force * self.t_sampling

        self.update_observation()

        barrier = (self.state[0] - 2 * self.radius_obst) ** 2 - np.log(
            max(self.state[0], 1e-6)
        )
        reward = max(0, -self.state[1] - 0.1 * barrier + 3.5)

        self.steps_left -= 1
        self.check_terminate()

        done = self.steps_left <= 0 or self.terminate
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.terminate = False
        self.x, self.y = self.state_init[0], self.state_init[1]
        if self.random_init:
            self.x = np.float32(
                self.state_init[0] + self.spread[0] * np.random.uniform(0, 1)
            )
            self.y = np.float32(
                self.state_init[1] + self.spread[1] * np.random.uniform(-1.0, 1.0)
            )
            if self.hybridlearning:
                while not self.M_ext.in_M_ext(
                    np.array([self.x, self.y], dtype=np.float32)
                ):
                    self.x = np.float32(
                        self.state_init[0] + self.spread[0] * np.random.uniform(0, 1)
                    )
                    self.y = np.float32(
                        self.state_init[1]
                        + self.spread[1] * np.random.uniform(-1.0, 1.0)
                    )
        self.update_observation()
        self.steps_left = self.steps
        return self.state
