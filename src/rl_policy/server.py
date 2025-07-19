import asyncio
import os
import signal
import numpy as np
import matplotlib.pyplot as plt
from grpclib.server import Server
from grpclib.exceptions import GRPCError
from grpclib.const import Status
from dotenv import load_dotenv

from stable_baselines3 import DQN
from typing import List

from scipy.interpolate import splprep, splev


from hyrl_api import obstacle_avoidance_grpc
from hyrl_api import obstacle_avoidance_pb2 as oa_proto
from HyRL.utils import ObstacleAvoidance
from dataclasses import dataclass
from pathlib import Path


import numpy as np
from numpy import linalg as LA
from HyRL.obstacleavoidance_env import ObstacleAvoidance, BBox, Point, Obstacle
from HyRL.utils import (
    find_critical_points,
    state_to_observation_OA,
    get_state_from_env_OA,
    find_X_i,
    train_hybrid_agent,
    M_i,
    M_ext,
    HyRL_agent,
    state_to_observation,
    simulate_obstacleavoidance,
    visualize_M_ext,
)
from pathlib import Path

from importlib.resources import path
import importlib.resources as pkg_resources


@dataclass
class ObstacleAvoidanceModels:
    hybrid_q0: HyRL_agent
    hybrid_q1: HyRL_agent
    standard: DQN


def initialize_hybrid_models():
    # Load pre-trained models
    with Path("HyRL", "models", "dqn_obstacleavoidance") as model_dir:
        model_path = Path(model_dir)
        print(f"Loading main model from {model_path}")
        model = DQN.load(str(model_path))

    with Path("HyRL", "models", "dqn_obstacleavoidance_0") as a0_dir:
        a0_path = Path(a0_dir)
        print(f"Loading agent_0 from {a0_path}")
        agent_0 = DQN.load(str(a0_path))

    with Path("HyRL", "models", "dqn_obstacleavoidance_1") as a1_dir:
        a1_path = Path(a1_dir)
        print(f"Loading agent_1 from {a1_path}")
        agent_1 = DQN.load(str(a1_path))

    print("✅ Successfully loaded pre-trained models")

    # Loading in the trained agent
    model = DQN.load(Path("HyRL/models") / "dqn_obstacleavoidance")
    bounds = BBox(x_min=0.0, x_max=3.0, y_min=-1.5, y_max=1.5)
    obstacle = Obstacle(center=Point(x=1.5, y=0.0), r=0.75)
    goal = Point(x=3.0, y=0.0)

    # finding the set of critical points
    #
    M_star = find_critical_points(
        30,
        bounds,
        obstacle,
        goal,
        model,
        environent=ObstacleAvoidance,
        custom_state_to_observation=state_to_observation(obstacle, goal),
        get_state_from_env=get_state_from_env_OA,
    )
    if M_star is None:
        raise ValueError("No critical points found. Please check your parameters.")
    M_star = M_star[np.argsort(M_star[:, 0])]

    # building sets M_0 and M_1
    M_0 = M_i(M_star, index=0)
    M_1 = M_i(M_star, index=1)

    # finding the extension sets
    X_0 = find_X_i(M_0, model)
    X_1 = find_X_i(M_1, model)
    M_ext0 = M_ext(M_0, X_0)
    M_ext1 = M_ext(M_1, X_1)

    # # visualizing the extended sets
    # visualize_M_ext(M_ext0, figure_number=1)
    # visualize_M_ext(M_ext1, figure_number=2)

    hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=0)
    hybrid_agent_q1 = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=1)

    return ObstacleAvoidanceModels(
        hybrid_q0=hybrid_agent, hybrid_q1=hybrid_agent_q1, standard=model
    )


class DroneService(obstacle_avoidance_grpc.ObstacleAvoidanceServiceBase):
    direction_map = {
        0: oa_proto.HeadingDirection.STRAIGHT,  # 1
        1: oa_proto.HeadingDirection.LEFT,  # 2
        2: oa_proto.HeadingDirection.HARD_LEFT,  # 3
        3: oa_proto.HeadingDirection.RIGHT,  # 4
        4: oa_proto.HeadingDirection.HARD_RIGHT,  # 5
    }

    def __init__(self, models: ObstacleAvoidanceModels):
        self.hybrid_agent_q0 = models.hybrid_q0
        self.hybrid_agent_q1 = models.hybrid_q1
        self.agent = models.standard

    def agent_select(
        self, state: oa_proto.DroneState, model_type: oa_proto.ModelType.ValueType
    ) -> HyRL_agent | DQN:
        if model_type == oa_proto.ModelType.STANDARD:
            agent = self.agent
        else:
            # agent = self.hybrid_agent_q0 if state.y > 0 else self.hybrid_agent_q1
            agent = self.hybrid_agent_q1
        return agent

    async def GetDirection(self, stream):
        request: oa_proto.DirectionRequest = await stream.recv_message()
        print(f"Received drone state: {request}")

        state = np.array([request.state.x, request.state.y])
        obs = state_to_observation_OA(state)

        agent = self.agent_select(request.state, request.model_type)
        action_array, _ = agent.predict(obs)
        if isinstance(action_array, np.ndarray):
            action = int(action_array.item())
        else:
            action = int(action_array)

        # Send response
        response = oa_proto.DirectionResponse(
            discrete_heading=oa_proto.DiscreteHeading(
                direction=self.direction_map.get(
                    action, oa_proto.HeadingDirection.STRAIGHT
                )
            )
        )
        print(f"Response direction: {response.discrete_heading}")
        await stream.send_message(response)

    async def GetTrajectory(self, stream):
        def smooth_path(states, num_waypoints):
            if len(states) < 3:
                return states
            # Fit a spline to the states
            tck, u = splprep(np.array(states).T, s=2)
            new_u = np.linspace(0, 1, num_waypoints)
            smoothed_states = splev(new_u, tck)
            return np.array(smoothed_states).T.tolist()

        request: oa_proto.TrajectoryRequest = await stream.recv_message()
        print(f"Received trajectory request: {request}")
        if 0 < request.num_waypoints < 3:
            raise GRPCError(
                Status.INVALID_ARGUMENT,
                "num_waypoints must be greater than or equal to 3, or less than 0",
            )

        [x_start, y_start, z_start] = [
            request.state.x,
            request.state.y,
            request.state.z,
        ]
        [x_target, y_target, z_target] = [
            request.target_state.x,
            request.target_state.y,
            request.target_state.z,
        ]
        state = np.array([x_start, y_start], dtype=np.float32)
        print(f"Initial state: {state}, Target state: {[x_target, y_target]}")
        states: List[List[float]] = [list(state) + [z_start]]
        duration_s = request.duration_s

        # Agent select
        # The q0 agent will bias to go up and around the obstacle, while q1 will bias to go the other way around
        agent = self.agent_select(request.state, request.model_type)

        # t_sampling = request.sampling_time if request.sampling_time else 0.05
        # steps = int(duration_s / t_sampling)
        # env = ObstacleAvoidance(
        #     state_init=state, random_init=False, steps=steps, t_sampling=t_sampling
        # )

        # target_pos = np.array([x_target, y_target])
        # dist_threshold = 0.1

        print(f"Simulating obstacle avoidance with agent: {agent}")

        orig, hy, switch = simulate_obstacleavoidance(
            self.hybrid_agent_q1, self.agent, state
        )

        print(f"Original path: {orig}")
        print(f"Hybrid path: {hy}")

        total_states = len(hy)
        states = smooth_path(hy, request.num_waypoints)
        print(
            f"Generated trajectory with {total_states} states, smoothed to {len(states)} waypoints"
        )
        print(states)
        response = oa_proto.TrajectoryResponse(
            trajectory=[
                oa_proto.DroneState(x=x, y=y, z=request.state.z) for x, y in states
            ]
        )
        print(f"Total states generated: {total_states}")
        await stream.send_message(response)


async def main():
    # Initialize the hybrid agent at startup
    print("Initializing RL models...")
    hybrid_agent = initialize_hybrid_models()

    server = Server([DroneService(hybrid_agent)])
    await server.start("127.0.0.1", 50051)
    print("gRPC server running on 127.0.0.1:50051")

    # Set up signal handling
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def shutdown():
        print("\nShutting Down Server...")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown)

    # Wait for Ctrl+C shutdown signal
    await stop.wait()
    server.close()
    await server.wait_closed()
    print("Server Stopped.")


if __name__ == "__main__":
    load_dotenv()
    print("Starting gRPC server...")
    asyncio.run(main())
