import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os

combined_xml_path = "humanoid.xml"  # Path to the combined XML file containing humanoid, ball, and goal


class HumanoidSoccerEnv3D(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode='human', combined_xml_path=combined_xml_path):
        super(HumanoidSoccerEnv3D, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)  # Humanoid joints
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(376,), dtype=np.float32)  # Humanoid observation space

        # Load Mujoco model
        if not os.path.exists(combined_xml_path):
            raise FileNotFoundError(f"Combined XML file not found at: {combined_xml_path}")

        # Load the combined model
        self.model = mujoco.MjModel.from_xml_path(combined_xml_path)
        self.data = mujoco.MjData(self.model)

        # Viewer setup for rendering
        self.render_mode = render_mode
        self.viewer = None
        if render_mode == 'human':
            self.viewer = mujoco.viewer.launch(self.model, self.data)

        # Set initial positions for ball and goal
        self.soccer_ball_position = np.array([0.0, 1.0, 0.1])
        self.goal_position = np.array([5.0, 0.0, 0.0])
        self.goal_radius = 1.0

        # Set positions in the model data using xpos
        ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.data.xpos[ball_body_id][:3] = self.soccer_ball_position
        self.data.xpos[goal_body_id][:3] = self.goal_position

        # Initialize the state
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.done = False

        # Reset the positions of all entities
        self.data.qpos[:3] = [0.0, 0.0, 1.4]  # Reset humanoid to initial position (slightly above ground)
        self.soccer_ball_position = np.array([0.0, 1.0, 0.1])  # Reset ball position

        # Update positions using xpos
        ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.data.xpos[ball_body_id][:3] = self.soccer_ball_position

        # Return the initial observation
        return self._get_obs(), {}

    def step(self, action):
        # Apply action to the humanoid
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Calculate distance to soccer ball
        humanoid_position = self.data.qpos[:3]
        distance_to_ball = np.linalg.norm(humanoid_position[:2] - self.soccer_ball_position[:2])

        # Reward system
        reward = 0
        if distance_to_ball <= 0.5:  # Kick the ball if close enough
            direction_to_goal = self.goal_position[:2] - self.soccer_ball_position[:2]
            self.soccer_ball_position[:2] += direction_to_goal * 0.2  # Move ball toward goal
            reward += 1  # Reward for kicking

        # Check if ball reaches the goal
        if np.linalg.norm(self.soccer_ball_position[:2] - self.goal_position[:2]) < self.goal_radius:
            reward += 10  # Goal reward
            self.done = True  # End episode when goal is scored

        # Create observation
        obs = self._get_obs()

        # Render if necessary
        if self.render_mode == 'human' and self.viewer is not None:
            self.viewer.render()
            
        # Return step information
        return obs, reward, self.done, False, {}

    def _get_obs(self):
        """Return the current observation."""
        # Include humanoid position, velocities, and other observations
        obs = np.concatenate([
            self.data.qpos.flat[:],  # Generalized positions
            self.data.qvel.flat[:],  # Generalized velocities
        ])
        return obs

    def render(self, mode='human'):
        if mode == 'human' and self.viewer is not None:
            self.viewer.render()

    def close(self):
        if self.viewer is not None:
            mujoco.viewer.close(self.viewer)
            self.viewer = None


# Instantiate the 3D environment with the path to your combined Mujoco XML file
env = HumanoidSoccerEnv3D(render_mode='human', combined_xml_path=combined_xml_path)

# Run a random policy for testing
obs, _ = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()
