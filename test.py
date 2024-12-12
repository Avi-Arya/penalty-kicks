from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from soccer_env import HumanoidSoccerEnv  # Import your custom environment
from walker_demo import Walker2dEnv

# Load the trained model
model = PPO.load("./models/ppo_humanoid_soccer_final")
from stable_baselines3.common.env_util import make_vec_env
def create_env(render_mode="rgb_array"):
    """
    Environment creation with potential curriculum learning
    """
    env = Walker2dEnv(render_mode=render_mode)
    return env
# Create the environment
env = make_vec_env(create_env, n_envs=1)
print("Environment initialized!")

obs = env.reset()
import time
# Loop for evaluation
for _ in range(10000):
    # Ensure observation is properly shaped
    obs = np.pad(obs.flatten(), (0, 4), mode='constant', constant_values=0)
    obs = np.expand_dims(obs, axis=0)  # Add batch dimension
    print("Observation shape:", obs.shape)

    # Predict action
    action, _ = model.predict(obs, deterministic=True)

    # Debug original action
    print("Original action shape:", action.shape)

    # Ensure correct shape and bounds
    action = action.squeeze()[:6]  # Remove batch dimension and truncate
    action = np.clip(action, env.action_space.low, env.action_space.high)
    print("Corrected action shape:", action.shape)

    # Pass to the environment
    obs, reward, terminated, truncated, info = env.step(action)


    if terminated or truncated:
        print("Terminating episode!")
        obs, info = env.reset()

