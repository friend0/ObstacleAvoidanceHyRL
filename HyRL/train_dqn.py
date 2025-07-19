import os
import torch as th
import matplotlib.pyplot as plt
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

load = True
train = False
save = False

# Create log dir
log_dir = "tmp/hyrl"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment with normalization
env = DummyVecEnv(
    [lambda: RecordEpisodeStatistics(ObstacleAvoidance(render_mode="rgb_array"))]
)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecVideoRecorder(
    env,
    video_folder="videos/",
    record_video_trigger=lambda step: step % 100_000 == 0,
    video_length=300,  # number of steps to record
    name_prefix="dqn-obstacleavoid",
)

if load:
    model = DQN.load(MODELS / "dqn_obstacleavoidance", env)
else:
    # building the model
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64])
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=0.0001946,
        gamma=0.9664,
    )

    model = DQN(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
        # learning_rate=2.5e-4,
        # gamma=0.9664,
        # buffer_size=125_000,
        # batch_size=64,
        # exploration_fraction=0.2,
        # exploration_final_eps=0.05,
        # train_freq=4,
        # target_update_interval=500,
        # tau=1.0,
        # learning_starts=25_000,
        # optimize_memory_usage=True,
    )

# Training configuration
# timesteps = 3_000_000
timesteps = 1_000_000
if train:
    # Separate evaluation env
    eval_env = ObstacleAvoidance()

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=96, verbose=1)
    eval_callback = EvalCallback(
        eval_env, callback_on_new_best=callback_on_best, verbose=1
    )
    timesteps = 1000000
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    if save:
        model.save("dqn_obstacleavoidance")

plot_results(
    [log_dir], timesteps, results_plotter.X_TIMESTEPS, "DQN Obstacle Avoidance"
)
plt.show()
