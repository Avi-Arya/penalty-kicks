import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

class HumanoidSoccerEnv3D(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode='human', humanoid_urdf_path="humanoid3.urdf"):
        super(HumanoidSoccerEnv3D, self).__init__()

        self.render_mode = render_mode


        # Initialize PyBullet
        self.physics_client = p.connect(p.GUI if render_mode == 'human' else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Load ground plane
        p.loadURDF("plane.urdf")

        # Initialize object positions
        self.humanoid_position = np.array([0.0, 0.0, 1.9])  # Adjusted starting height for humanoid
        self.soccer_ball_position = np.array([0.0, 1.0, 0.1])  # Place ball slightly above ground
        self.goal_position = np.array([5.0, 0.0, 3])  # Define goal position

        # Load ball, humanoid (from local URDF), and goal visualization
        if os.path.exists(humanoid_urdf_path):
            # Load the humanoid model with orientation and scaling adjustments
            self.humanoid_id = p.loadURDF(
                humanoid_urdf_path,
                self.humanoid_position,
                p.getQuaternionFromEuler([np.pi / 2, 0, 0]),  # Rotate upright if sideways
                globalScaling=0.5  # Adjust scaling to match ball size
            )
        else:
            raise FileNotFoundError(f"Humanoid URDF not found at: {humanoid_urdf_path}")
        
        # Extract joint information
        self.joint_indices = [i for i in range(p.getNumJoints(self.humanoid_id)) if p.getJointInfo(self.humanoid_id, i)[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.joint_indices)

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6 + self.num_joints,), dtype=np.float32)

        
        self.ball_id = p.loadURDF("sphere2.urdf", self.soccer_ball_position, globalScaling=0.5)  # Small ball
        

        # Scale the goal appropriately to fit the environment
        self.goal_scaling = 2.0  # Adjusted scaling factor
        self.goal_id = p.loadURDF("goal.urdf", self.goal_position, globalScaling=self.goal_scaling)  # Goal marker

        self.goal_radius = 1.0 * self.goal_scaling  # Adjust radius for goal size
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset positions
        self.humanoid_position = np.array([0.0, 0.0, 1.9])  # Adjusted height for humanoid
        self.soccer_ball_position = np.array([0.0, 1.0, 0.1])
        p.resetBasePositionAndOrientation(self.humanoid_id, self.humanoid_position, p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
        p.resetBasePositionAndOrientation(self.ball_id, self.soccer_ball_position, [0, 0, 0, 1])

         # Reset joint positions to neutral
        for joint_index in self.joint_indices:
            p.resetJointState(self.humanoid_id, joint_index, targetValue=0.0, targetVelocity=0.0)


        # Return the initial observation
        return self._get_obs(), {}

    def step(self, action):
        # Update humanoid position based on action
        # self.humanoid_position[:2] += action * 0.1  # Scale movement in x and y
        p.resetBasePositionAndOrientation(self.humanoid_id, self.humanoid_position, p.getQuaternionFromEuler([np.pi / 2, 0, 0]))

        # Calculate distance to soccer ball
        distance_to_ball = np.linalg.norm(self.humanoid_position[:2] - self.soccer_ball_position[:2])

        # Apply actions to humanoid joints
        for i, joint_index in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.humanoid_id,
                jointIndex=joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=action[i] * 50  # Scale torque
            )

        
        # Reward system
        reward = 0
        if distance_to_ball <= 0.5:  # Kick the ball if close enough
            direction_to_goal = self.goal_position[:2] - self.soccer_ball_position[:2]
            self.soccer_ball_position[:2] += direction_to_goal * 0.2  # Move ball toward goal
            p.resetBasePositionAndOrientation(self.ball_id, self.soccer_ball_position, [0, 0, 0, 1])
            reward += 1  # Reward for kicking

        # Check if ball reaches the goal
        if np.linalg.norm(self.soccer_ball_position[:2] - self.goal_position[:2]) < self.goal_radius:
            reward += 10  # Goal reward
            self.done = True  # End episode when goal is scored

        # Step the simulation for visualization
        p.stepSimulation()
        time.sleep(1 / self.metadata['render_fps'])  # Control simulation speed

        # Create observation
        obs = self._get_obs()

        # Return step information
        return obs, reward, self.done, False, {}

    def _get_obs(self):
        """Return the current observation."""
        joint_states = np.array([p.getJointState(self.humanoid_id, i)[0] for i in self.joint_indices])
        return np.concatenate((self.humanoid_position[:2], self.soccer_ball_position[:2], self.goal_position[:2]))

    def render(self):
        """PyBullet handles real-time rendering automatically in GUI mode."""
        pass

    def close(self):
        p.disconnect()


# Instantiate the 3D environment with the path to your humanoid URDF
env = HumanoidSoccerEnv3D(render_mode='human', humanoid_urdf_path="humanoid3.urdf")

# Run a random policy for testing
obs, _ = env.reset()
for _ in range(1000000):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()

env.close()