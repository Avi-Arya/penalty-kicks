import numpy as np
from enviorn2 import HumanoidSoccerEnv

# Create the custom humanoid soccer environment
env = HumanoidSoccerEnv(render_mode="human")

# Reset the environment
obs, info = env.reset()

# Random action loop for visualization
for _ in range(1000):  # Simulate 1000 steps
    action = env.action_space.sample()  # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
