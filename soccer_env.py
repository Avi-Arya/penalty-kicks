import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
import os
from pathlib import Path

class HumanoidSoccerEnv(HumanoidEnv):
    def __init__(self, render_mode="human"):
        # First, get the path to the original humanoid XML
        gym_path = Path(gym.__file__).parent
        xml_file = gym_path / "envs/mujoco/assets/humanoid.xml"
        
        # Read and modify the XML to include our additions
        with open(xml_file, 'r') as f:
            xml_contents = f.read()
        
        # Add the ball and goal before the closing worldbody tag
        ball_and_goal = f"""
            <body name="ball" pos="3 0 0.11">
                <joint name="ball_free" type="free"/>
                <geom name="ball" type="sphere" size="0.11" rgba="1 0 0 1" mass="0.45" friction="0.8" condim="4"/>
            </body>
            <body name="goal" pos="6 0 0">
                <geom name="crossbar" type="capsule" fromto="0 -1.5 2 0 1.5 2" 
                    size="0.05" rgba="0 0 1 1" contype="2" conaffinity="2"/>
                <geom name="post1" type="capsule" fromto="0 -1.5 0 0 -1.5 2" 
                    size="0.05" rgba="0 0 1 1" contype="2" conaffinity="2"/>
                <geom name="post2" type="capsule" fromto="0 1.5 0 0 1.5 2" 
                    size="0.05" rgba="0 0 1 1" contype="2" conaffinity="2"/>
            </body>
        """
        
        # Add ground friction modification and make the ground visible
        modified_xml = xml_contents.replace(
            '<geom name="floor" type="plane"',
            '<geom name="floor" type="plane" rgba="0.5 0.5 0.5 1" friction="1.0 0.5 0.5"'
        )
        
        # Insert ball and goal
        modified_xml = modified_xml.replace('</worldbody>', f'{ball_and_goal}</worldbody>')
        
        # Save the modified XML to a temporary file
        self.temp_path = r"C:\Users\sahil\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\gymnasium\envs\mujoco\assets\temp_humanoid_soccer.xml"
        with open(self.temp_path, 'w') as f:
            f.write(modified_xml)

        MujocoEnv.__init__(
            self,
            self.temp_path,
            5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            ),
            render_mode=render_mode
        )
        
        # Define positions and sizes
        self.ball_pos = np.array([3.0, 0.0, 0.11])
        self.goal_pos = np.array([6.0, 0.0, 1.0])
        self.goal_size = np.array([3.0, 0.0, 2.0])
        
        # Update state spaces to include ball
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = np.zeros_like(self.data.qvel)
        
        # Adjust initial humanoid position to be further from ball
        self.init_qpos[0] = 0.0  # x position
        self.init_qpos[1] = 0.0  # y position
        self.init_qpos[2] = 1.0  # z position
        
        # Set initial ball position
        self.init_qpos[-7:-4] = self.ball_pos  # xyz position
        self.init_qpos[-4:] = [1, 0, 0, 0]    # quaternion orientation
        
        # Extend the observation space to include ball information
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.observation_space.shape[0] + 6,),
            dtype=np.float64
        )

        # Initialize reset noise scale
        self._reset_noise_scale = 1e-2

        self._exclude_current_positions_from_observation = True
        
        
        self._forward_reward_weight = 1.25
        self._ctrl_cost_weight = 0.1
        self._healthy_reward = 5.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range = (1.0, 2.0)


    def __del__(self):
        # Clean up the temporary file
        if hasattr(self, 'temp_path'):
            try:
                os.remove(self.temp_path)
            except:
                pass

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        # Reset humanoid with less noise to reduce spazzing
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        
        # Add small noise to humanoid position and orientation
        qpos[:7] += self.np_random.uniform(low=noise_low, high=noise_high, size=7) * 0.1
        
        # Reset ball position with small random variation
        ball_pos = self.ball_pos + self.np_random.uniform(-0.1, 0.1, size=3)
        qpos[-7:-4] = ball_pos
        qpos[-4:] = [1, 0, 0, 0]  # ball orientation
        
        # Reset velocities with small noise
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        ) * 0.1
        
        # Ensure ball velocity is zero initially
        qvel[-6:] = 0
        
        self.set_state(qpos, qvel)
        
        return self._get_obs()

    # def _get_obs(self):
    #     position = self.data.qpos.flat.copy()
    #     velocity = self.data.qvel.flat.copy()

    #     com_inertia = self.data.cinert.flat.copy()
    #     com_velocity = self.data.cvel.flat.copy()

    #     actuator_forces = self.data.qfrc_actuator.flat.copy()
    #     external_contact_forces = self.data.cfrc_ext.flat.copy()

    #     # Get ball position and velocity
    #     ball_xyz = self.data.qpos[-7:-4]
    #     ball_vel = self.data.qvel[-6:-3]

    #     # Calculate relative position between humanoid and ball
    #     humanoid_pos = self.data.qpos[:3]
    #     relative_ball_pos = ball_xyz - humanoid_pos

    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]

    #     observations = np.concatenate((
    #         position, velocity, com_inertia, com_velocity,
    #         actuator_forces, external_contact_forces,
    #         relative_ball_pos, ball_vel
    #     ))

    #     return observations
    
    def _get_obs(self):
    # Extract the original observations from the parent class
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        # Ball position and velocity
        ball_pos = self.data.qpos[-7:-4]  # Ball x, y, z position
        ball_vel = self.data.qvel[-6:-3]  # Ball x, y, z velocity

        # Humanoid position and velocity
        humanoid_pos = self.data.qpos[:3]  # Humanoid x, y, z position
        humanoid_vel = self.data.qvel[:3]  # Humanoid x, y, z velocity

        # Foot angles (example: joint angles for feet, customize as needed)
        foot_angles = self.data.qpos[7:9]  # Assuming feet are indexed here

        # Relative positions
        humanoid_to_ball = ball_pos - humanoid_pos  # Vector from humanoid to ball
        ball_to_goal = self.goal_pos - ball_pos     # Vector from ball to goal

        # Combine all features into the observation
        additional_features = np.concatenate((
            ball_pos,       # Ball position
            ball_vel,       # Ball velocity
            humanoid_pos,   # Humanoid position
            humanoid_vel,   # Humanoid velocity
            foot_angles,    # Foot angles
            humanoid_to_ball, # Vector from humanoid to ball
            ball_to_goal    # Vector from ball to goal
        ))

        # Combine the original observations with the additional features
        observations = np.concatenate((
            position, velocity, com_inertia, com_velocity,
            actuator_forces, external_contact_forces,
            additional_features
        ))

        return observations


    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        
        # Get ball position before
        ball_pos_before = self.data.qpos[-7:-4].copy()
        
        # Step
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Calculate reward
        xy_position_after = self.data.qpos[0:2].copy()
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        
        ball_pos_after = self.data.qpos[-7:-4].copy()
        ball_velocity = (ball_pos_after - ball_pos_before) / self.dt
        
        # Add ball-related rewards
        dist_to_ball = np.linalg.norm(xy_position_after - ball_pos_after[:2])
        ball_to_goal = np.linalg.norm(ball_pos_after[:2] - self.goal_pos[:2])
        
        reward -= 0.05 * dist_to_ball  # Small penalty for being far from ball
        reward -= 0.05 * ball_to_goal  # Small penalty for ball being far from goal
        
        # Bonus for ball moving towards goal
        if ball_velocity[0] > 0:  # Ball moving towards goal
            reward += 0.1 * ball_velocity[0]
        
        # Big reward for scoring
        if (ball_pos_after[0] > self.goal_pos[0] and 
            abs(ball_pos_after[1]) < 1.5 and 
            ball_pos_after[2] < 2.0):
            reward += 100.0
            terminated = True
        
        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    # Create and register the environment
    env = HumanoidSoccerEnv(render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    
    try:
        for _ in range(10000):
            # Random action
            action = env.action_space.sample()
            
            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
    finally:
        env.close()