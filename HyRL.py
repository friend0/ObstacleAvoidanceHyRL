import numpy as np
from numpy import linalg as LA
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from utils import (
    find_critical_points,
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


@dataclass
class State:
    x: float
    y: float


class AgentSelect(Enum):
    agent_0 = 0
    agent_1 = 1


class ObstacleAvoidancePlanner:
    def __init__(
        self,
        dqn="dqn_obstacleavoidance",
        q: AgentSelect = AgentSelect.agent_0,
        resolution=30,
        x_range=(0, 3),
        y_range=(-1.5, 1.5),
        visualize=False,
    ) -> None:
        self.model = DQN.load(dqn)
        self.resolution = resolution
        self.state_difference = LA.norm(
            np.array([x_range[1] - x_range[0], y_range[1] - y_range[0]])
        )
        self.initial_points = []
        for idx in range(resolution):
            for idy in range(resolution):
                self.initial_points.append(
                    np.array([x_range[idx], y_range[idy]], dtype=np.float32)
                )

        self.m_star = find_critical_points(
            self.initial_points,
            self.state_difference,
            self.model,
            ObstacleAvoidance,
            min_state_difference=1e-2,
            steps=5,
            threshold=1e-1,
            n_clusters=8,
            custom_state_to_observation=state_to_observation_OA,
            get_state_from_env=get_state_from_env_OA,
            verbose=False,
        )
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

        agent_0 = DQN.load("dqn_obstacleavoidance_0")
        agent_1 = DQN.load("dqn_obstacleavoidance_1")
        self.hybrid_agent = HyRL_agent(agent_0, agent_1, m_ext0, m_ext1, q_init=q.value)

    def control(self, state: State):
        action_hyb, switch = self.hybrid_agent.predict(state_hyb + disturbance)
        env_hyb.state = state_hyb
        _, reward_hyb, done, _ = env_hyb.step(action_hyb)
        state_hyb = get_state_from_env_OA(env_hyb)
