import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from HyRL.obstacleavoidance_env import ObstacleAvoidance, BBox, Point, Obstacle
from stable_baselines3 import DQN
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

if __name__ == "__main__":
    # Loading in the trained agent
    model = DQN.load(Path("src/HyRL/models") / "dqn_obstacleavoidance")
    bounds = BBox(x_min=0.0, x_max=3.0, y_min=-1.5, y_max=1.5)
    obstacle = Obstacle(center=Point(x=1.5, y=0.0), r=0.75)
    goal = Point(x=3.0, y=0.0)

    # finding the set of critical points

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

    import pickle

    # Pack your variables into a dictionary
    # np.savez("critical_points.npz", M_star=M_star)

    # visualizing the extended sets
    print("Visualizing the extended sets...")
    visualize_M_ext(M_ext0, figure_number=1)
    visualize_M_ext(M_ext1, figure_number=2)

    # building the environment for hybrid learning
    env_0 = ObstacleAvoidance(
        hybridlearning=True, M_ext=M_ext0, bounds=bounds, obstacle=obstacle, goal=goal
    )
    env_1 = ObstacleAvoidance(
        hybridlearning=True, M_ext=M_ext1, bounds=bounds, obstacle=obstacle, goal=goal
    )

    # training the new agents
    training2 = False
    if training2:
        for radius in [0.25, 0.50]:
            agent_0 = train_hybrid_agent(
                env_0,
                load_agent="dqn_obstacleavoidance",
                save_name=f"dqn_obstacleavoidance_0_{radius * 100}",
                M_exti=M_ext0,
                timesteps=300000,
            )
            agent_1 = train_hybrid_agent(
                env_1,
                load_agent="dqn_obstacleavoidance",
                save_name=f"dqn_obstacleavoidance_1_{radius * 100}",
                M_exti=M_ext1,
                timesteps=300000,
            )
    else:
        agent_0 = DQN.load(Path("src/HyRL/models") / "dqn_obstacleavoidance_0")
        agent_1 = DQN.load(Path("src/HyRL/models") / "dqn_obstacleavoidance_1")

    # simulation the hybrid agent compared to the original agent
    starting_conditions = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.055], dtype=np.float32),
        np.array([0.0, -0.055], dtype=np.float32),
        np.array([0.0, 0.15], dtype=np.float32),
        np.array([0.0, -0.15], dtype=np.float32),
        np.array([0.0, 0.25], dtype=np.float32),
        np.array([0.0, -0.25], dtype=np.float32),
    ]
    for q in range(2):
        print(f"Simulating for q = {q}")
        for state_init in starting_conditions:
            print(f"Starting condition: {state_init}")
            hybrid_agent = HyRL_agent(agent_0, agent_1, M_ext0, M_ext1, q_init=q)
            simulate_obstacleavoidance(
                hybrid_agent, model, state_init, figure_number=3 + q
            )
        save_name = "OA_HyRLDQN_Sim_q" + str(q) + ".png"
        plt.savefig(save_name, format="png")
