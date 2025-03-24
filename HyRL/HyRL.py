import HyRL
import numpy as np
from numpy import linalg as LA
from HyRL.obstacleavoidance_env import ObstacleAvoidance, BBox, Obstacle, Point, State
from stable_baselines3 import DQN
from HyRL.utils import (
    find_critical_points,
    state_to_observation,
    state_to_observation_OA,
    get_state_from_env_OA,
    find_X_i,
    M_i,
    M_ext,
    HyRL_agent,
    visualize_M_ext,
)
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AgentSelect(Enum):
    agent_0 = 0
    agent_1 = 1


class ObstacleAvoidancePlanner:
    MODELS = Path("HyRL/models")

    def __init__(
        self,
        dqn="dqn_obstacleavoidance",
        q: AgentSelect = AgentSelect.agent_0,
        resolution=30,
        bounds: BBox = BBox(x_min=0.0, x_max=3.0, y_min=-1.5, y_max=1.5),
        obstacle: Obstacle = Obstacle(center=Point(x=1.5, y=0.0), r=0.75),
        goal: Point = Point(x=3.0, y=0.0),
        visualize=False,
    ) -> None:
        self.resolution = resolution
        self.bounds = bounds
        self.obstacle = obstacle
        self.goal = goal
        self.model = DQN.load(self.MODELS / dqn)

        x_ = np.linspace(self.bounds.x_min, self.bounds.x_max, resolution)
        y_ = np.linspace(self.bounds.y_min, self.bounds.y_max, resolution)

        self.state_difference = LA.norm(np.array([x_[1] - x_[0], y_[1] - y_[0]]))
        self.initial_points = []
        for idx in range(resolution):
            for idy in range(resolution):
                self.initial_points.append(
                    np.array([x_[idx], y_[idy]], dtype=np.float32)
                )

        self.m_star = find_critical_points(
            30,
            self.bounds,
            self.obstacle,
            self.goal,
            self.model,
            environent=ObstacleAvoidance,
            custom_state_to_observation=state_to_observation(obstacle, goal),
            get_state_from_env=get_state_from_env_OA,
        )
        if not self.m_star:
            raise ValueError("No critical points found. Please check your parameters.")

        # TODO: cleanup this sort
        self.m_star = self.m_star[np.argsort(self.m_star[:, 0])]

        # building sets M_0 and M_1
        m_0 = M_i(self.m_star, index=0)
        m_1 = M_i(self.m_star, index=1)

        # finding the extension sets
        x_0 = find_X_i(m_0, self.model)
        x_1 = find_X_i(m_1, self.model)
        m_ext0 = M_ext(m_0, x_0)
        m_ext1 = M_ext(m_1, x_1)

        if visualize:
            # visualizing the extended sets
            visualize_M_ext(m_ext0, figure_number=1)
            visualize_M_ext(m_ext1, figure_number=2)

        agent_0 = DQN.load(self.MODELS / "dqn_obstacleavoidance_0")
        agent_1 = DQN.load(self.MODELS / "dqn_obstacleavoidance_1")
        self.hybrid_agent = HyRL_agent(agent_0, agent_1, m_ext0, m_ext1, q_init=q.value)

    def control(self, state: State):
        action_hyb, switch = self.hybrid_agent.predict(state)
        env_hyb.state = state_hyb
        _, reward_hyb, done, _ = env_hyb.step(action_hyb)
        state_hyb = get_state_from_env_OA(env_hyb)
