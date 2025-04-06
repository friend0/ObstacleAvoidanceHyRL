import os
import torch as th
import matplotlib.pyplot as plt
from obstacleavoidance_env import ObstacleAvoidance
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.results_plotter import plot_results
from pathlib import Path
import numpy as np


# Hook into callback to store episode rewards
class RewardTrackerCallback(EvalCallback):
    def __init__(self, eval_env, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if len(self.locals["infos"]) > 0:
            info = self.locals["infos"][0]
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


MODELS = Path("HyRL/models")
load = False
train = True
save = True

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
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64])
    device = "mps" if th.backends.mps.is_available() else "cpu"

    model = DQN(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=2e-4,
        gamma=0.966,
        buffer_size=250_000,
        batch_size=256,
        exploration_fraction=0.20,
        exploration_final_eps=0.05,
        train_freq=4,
        target_update_interval=500,
        tau=1.0,
        learning_starts=25_000,
        # optimize_memory_usage=True,
    )

# Training configuration
# timesteps = 3_000_000
timesteps = 500_000
if train:
    eval_env = DummyVecEnv([lambda: RecordEpisodeStatistics(ObstacleAvoidance())])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=96, verbose=1)
    eval_callback = RewardTrackerCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback)

    if save:
        model.save(MODELS / "dqn_obstacleavoidance")

    window = 100
    smoothed = np.convolve(
        eval_callback.episode_rewards, np.ones(window) / window, mode="valid"
    )

    plt.plot(smoothed)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.grid(True)
    plt.show()

plot_results(
    [log_dir], timesteps, results_plotter.X_TIMESTEPS, "DQN Obstacle Avoidance"
)
plt.show()
