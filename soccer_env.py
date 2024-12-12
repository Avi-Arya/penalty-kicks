import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.spaces import Box
import os
from pathlib import Path
import random

class HumanoidSoccerEnv(HumanoidEnv):
    def __init__(self, render_mode="human"):
        import random

        # 1. Goal Dimensions
        self.goal_width = 8.0  # Total width (from -4 to 4 in y-direction)
        self.goal_height = 3.0  # Total height (from 0 to 3 in z-direction)
        self.grid_cols = 5
        self.grid_rows = 3
        self.region_width = self.goal_width / self.grid_cols
        self.region_height = self.goal_height / self.grid_rows

        # 2. Randomly Select a Region
        self.selected_row = random.randint(0, self.grid_rows - 1)  # 3 rows
        self.selected_col = random.randint(0, self.grid_cols - 1)  # 5 columns
        self.target_y = -4.0 + self.selected_col * self.region_width + self.region_width / 2
        self.target_z = 0.0 + self.selected_row * self.region_height + self.region_height / 2

        # 3. Ball Dimensions and Position
        self.ball_pos = np.array([2.0, 0.0, 0.12])  # Ball position
        self.ball_size = 0.12  # Radius of the ball

        # 4. Humanoid Dimensions and Position
        self.humanoid_pos = np.array([0.0, 0.0, 1.0])  # Humanoid starting position
        self.humanoid_size = np.array([1.0, 0.5, 1.8])  # Rough bounding box (for reference)

        # 5. Distance Between Ball and Humanoid
        self.distance_to_ball = np.linalg.norm(self.ball_pos - self.humanoid_pos)

        # Environment configuration
        self._forward_reward_weight = 1.25  # Weight for forward reward
        self._ctrl_cost_weight = 0.1  # Weight for control cost
        self._contact_cost_weight = 5e-7  # Weight for contact cost
        self._healthy_reward = 5.0  # Constant reward for being healthy
        self._terminate_when_unhealthy = True  # Terminate if unhealthy
        self._healthy_z_range = (1.0, 10.0)  # Range of z-coordinate for healthy humanoid
        self._reset_noise_scale = 1e-2  # Scale of random perturbations for initial state
        self._exclude_current_positions_from_observation = True  # Exclude x, y positions from observation




        # 6. Read Base XML
        gym_path = Path(gym.__file__).parent
        xml_file = gym_path / "envs/mujoco/assets/humanoid.xml"
        with open(xml_file, 'r') as f:
            xml_contents = f.read()

        # 7. Generate Modular XML
        ball_and_goal = f"""
            <body name="ball" pos="{self.ball_pos[0]} {self.ball_pos[1]} {self.ball_pos[2]}">
                <joint name="ball_free" type="free"/>
                <geom name="ball" type="sphere" size="{self.ball_size}" rgba="1 0 0 1" 
                    mass="0.45" friction="0.8" condim="4"/>
            </body>
            <body name="goal" pos="17.5 0 0">
                <geom name="crossbar" type="capsule" fromto="0 -{self.goal_width / 2} {self.goal_height} 
                    0 {self.goal_width / 2} {self.goal_height}" size="0.1" rgba="0 0 1 1" 
                    contype="2" conaffinity="2"/>
                <geom name="post1" type="capsule" fromto="0 -{self.goal_width / 2} 0 0 -{self.goal_width / 2} {self.goal_height}" 
                    size="0.1" rgba="0 0 1 1" contype="2" conaffinity="2"/>
                <geom name="post2" type="capsule" fromto="0 {self.goal_width / 2} 0 0 {self.goal_width / 2} {self.goal_height}" 
                    size="0.1" rgba="0 0 1 1" contype="2" conaffinity="2"/>
                <geom name="target" type="box" size="0.01 {self.region_width / 2} {self.region_height / 2}" 
                    pos="0 {self.target_y} {self.target_z}" rgba="1 1 0 0.5" contype="0" conaffinity="0"/>
            </body>
        """

        # Modify the XML to include the new components
        modified_xml = xml_contents.replace(
            '<geom name="floor" type="plane"',
            '<geom name="floor" type="plane" rgba="0.0 1.0 0.0 1" friction="1.0 0.5 0.5"'
        )
        modified_xml = modified_xml.replace('</worldbody>', f'{ball_and_goal}</worldbody>')

        # Save the modified XML to a temporary file
        self.temp_path = r"C:\Users\sahil\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\gymnasium\envs\mujoco\assets\temp_humanoid_soccer.xml"
        with open(self.temp_path, 'w') as f:
            f.write(modified_xml)

        # Initialize Mujoco Environment
        MujocoEnv.__init__(
            self,
            self.temp_path,
            5,
            observation_space=Box(
                low=-np.inf, high=np.inf, shape=(378,), dtype=np.float64
            ),
            render_mode=render_mode
        )

        # Define target region
        self.target = {
            "region": (self.selected_row, self.selected_col),
            "pos": (17.5, self.target_y, self.target_z),
            "size": (0.01, self.region_width / 2, self.region_height / 2)
        }

        # Observation Space Adjustment
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_space.shape[0] + 6,),
            dtype=np.float64
        )

        self._reset_noise_scale = 1e-2



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
        ball_pos = self.ball_pos + self.np_random.uniform(-0.01, 0.01, size=3)
        qpos[-7:-4] = ball_pos
        qpos[-4:] = [1, 0, 0, 0]  # ball orientation
        
        # Reset velocities with small noise
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        ) * 0.1
        
        # Ensure ball velocity is zero initially
        qvel[-6:] = 0
        
        self.selected_row = random.randint(0, self.grid_rows - 1)
        self.selected_col = random.randint(0, self.grid_cols - 1)
        self.target_y = -4.0 + self.selected_col * self.region_width + self.region_width / 2
        self.target_z = 0.0 + self.selected_row * self.region_height + self.region_height / 2
        self.target['region'] = (self.selected_row, self.selected_col)
        self.target['pos'] = (17.5, self.target_y, self.target_z)
        
        # Apply the state
        self.set_state(qpos, qvel)
        
        return self._get_obs()
    
    def _get_obs(self):
        # Extract the original observations from the parent class
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        com_inertia = self.data.cinert.flat.copy()
        com_velocity = self.data.cvel.flat.copy()

        actuator_forces = self.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.data.cfrc_ext.flat.copy()

        # Ball position and velocity (dynamically fetched from simulation)
        ball_pos = self.data.qpos[-7:-4].copy()  # Ball x, y, z position
        ball_vel = self.data.qvel[-6:-3].copy()  # Ball x, y, z velocity

        # Humanoid position and velocity (dynamically fetched from simulation)
        humanoid_pos = self.data.qpos[:3].copy()  # Humanoid x, y, z position
        humanoid_vel = self.data.qvel[:3].copy()  # Humanoid x, y, z velocity

        # Humanoid to ball vector
        humanoid_to_ball = ball_pos - humanoid_pos

        # Ball to target vector
        target_pos = np.array([self.target["pos"][0], self.target["pos"][1], self.target["pos"][2]])
        ball_to_target = target_pos - ball_pos

        # Combine all features into the observation
        additional_features = np.concatenate((
            ball_pos,          # Ball position
            ball_vel,          # Ball velocity
            humanoid_pos,      # Humanoid position
            humanoid_vel,      # Humanoid velocity
            humanoid_to_ball,  # Vector from humanoid to ball
            ball_to_target     # Vector from ball to target
        ))

        # Combine the original observations with the additional features
        observations = np.concatenate((
            position, velocity, com_inertia, com_velocity,
            actuator_forces, external_contact_forces,
            additional_features
        ))

        return observations




    def step(self, action):
        # Calculate goal position dynamically (center of the goal)
        goal_pos = np.array([17.5, 0.0, self.goal_height / 2])  # Center of the goal

        # Humanoid position before step (x, y)
        humanoid_pos_before = self.data.qpos[:2].copy()

        # Ball position before step (x, y, z)
        ball_pos_before = self.data.qpos[-7:-4].copy()

        # Step the simulation
        observation, reward, terminated, truncated, info = super().step(action)

        # Humanoid position after step (x, y)
        humanoid_pos_after = self.data.qpos[:2].copy()
        humanoid_velocity = (humanoid_pos_after - humanoid_pos_before) / self.dt

        # Ball position after step (x, y, z)
        ball_pos_after = self.data.qpos[-7:-4].copy()
        ball_velocity = (ball_pos_after - ball_pos_before) / self.dt

        # Calculate rewards
        dist_to_ball = np.linalg.norm(humanoid_pos_after - ball_pos_after[:2])
        ball_to_goal = np.linalg.norm(ball_pos_after[:2] - goal_pos[:2])

        reward -= 0.05 * dist_to_ball  # Penalize distance from ball
        reward -= 0.05 * ball_to_goal  # Penalize distance of ball from goal

        # Reward for ball moving towards goal
        if ball_velocity[0] > 0:  # Ball moving in positive x direction
            reward += 0.1 * ball_velocity[0]

        # Big reward for scoring
        if (ball_pos_after[0] > goal_pos[0] and 
            abs(ball_pos_after[1]) < self.goal_width / 2 and 
            ball_pos_after[2] < self.goal_height):
            reward += 100.0
            terminated = True

        # Update info dictionary with additional metrics
        info.update({
            "humanoid_position": humanoid_pos_after,
            "ball_position": ball_pos_after,
            "distance_to_ball": dist_to_ball,
            "distance_ball_to_goal": ball_to_goal,
            "ball_velocity": ball_velocity,
            "humanoid_velocity": humanoid_velocity,
        })

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