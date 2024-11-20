import gymnasium as gym
import numpy as np
import time

# Initialize the Gymnasium environment with real-time rendering
env = gym.make("Humanoid-v5", render_mode="human")

# Define the position of the soccer ball and the goal region
soccer_ball_position = np.array([1.0, 0.0, 0.0])  # Example initial position
goal_position = np.array([5.0, 0.0, 0.0])         # Example position for goal center
goal_radius = 1.0  # Radius defining the goal area

# Distance threshold to "kick" the ball
kick_distance_threshold = 0.5

# Simulate the environment using a random policy
def random_policy_simulation(steps=1000):
    global soccer_ball_position
    obs, _ = env.reset()

    for _ in range(steps):
        # Sample a random action from the action space
        action = env.action_space.sample()

        # Apply the random action and get results
        obs, _, done, _, _ = env.step(action)

        # Humanoid position in the environment
        humanoid_position = obs[:3]  # Assume the first 3 elements represent position
        
        # Calculate the distance between the humanoid and the soccer ball
        distance_to_ball = np.linalg.norm(humanoid_position - soccer_ball_position)

        # "Kick" the ball if within the threshold distance
        if distance_to_ball < kick_distance_threshold:
            # Move the ball towards the goal
            direction_to_goal = goal_position - soccer_ball_position
            soccer_ball_position += direction_to_goal * 0.1  # Adjust the multiplier for speed

        # Check if the soccer ball reached the goal
        if np.linalg.norm(soccer_ball_position - goal_position) < goal_radius:
            print("Goal scored!")
            soccer_ball_position = np.array([1.0, 0.0, 0.0])  # Reset ball position

        # Render the environment in real-time
        env.render()

        # Slow down for visualization
        time.sleep(0.01)  # Adjust the speed for visualization

        if done:
            obs, _ = env.reset()  # Reset if done

# Run the simulation with the random policy
random_policy_simulation(steps=1000)

# Close the environment
env.close()