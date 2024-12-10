import numpy as np
import gymnasium as gym
import random
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pathlib import Path
import time

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class Walker2dEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment builds on the hopper environment based on the work done by Erez, Tassa, and Todorov
    in ["Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks"](http://www.roboticsproceedings.org/rss07/p10.pdf)
    by adding another set of legs making it possible for the robot to walker forward instead of
    hop. Like other Mujoco environments, this environment aims to increase the number of independent state
    and control variables as compared to the classic control environments. The walker is a
    two-dimensional two-legged figure that consist of four main body parts - a single torso at the top
    (with the two legs splitting after the torso), two thighs in the middle below the torso, two legs
    in the bottom below the thighs, and two feet attached to the legs on which the entire body rests.
    The goal is to make coordinate both sets of feet, legs, and thighs to move in the forward (right)
    direction by applying torques on the six hinges connecting the six body parts.

    ### Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                 | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor      | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor        | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor       | -1          | 1           | foot_joint                       | hinge | torque (N m) |
    | 3   | Torque applied on the left thigh rotor | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
    | 4   | Torque applied on the left leg rotor   | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the left foot rotor  | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

    ### Observation Space

    Observations consist of positional values of different body parts of the walker,
    followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    By default, observations do not include the x-coordinate of the top. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will have 18 dimensions where the first dimension
    represent the x-coordinates of the top of the walker.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x-coordinate
    of the top will be returned in `info` with key `"x_position"`.

    By default, observation is a `ndarray` with shape `(17,)` where the elements correspond to the following:

    | Num | Observation                                      | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the top (height of hopper)       | -Inf | Inf | rootz (torso)                    | slide | position (m)             | 
    | 1   | angle of the top                                 | -Inf | Inf | rooty (torso)                    | hinge | angle (rad)              |
    | 2   | angle of the thigh joint                         | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 3   | angle of the leg joint                           | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 4   | angle of the foot joint                          | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 5   | angle of the left thigh joint                    | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 6   | angle of the left leg joint                      | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 7   | angle of the left foot joint                     | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |
    | 8   | velocity of the x-coordinate of the top          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | velocity of the z-coordinate (height) of the top | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angular velocity of the angle of the top         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the leg hinge                | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the foot hinge               | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the thigh hinge              | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the leg hinge                | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the foot hinge               | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
    ### Rewards
    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the walker is alive, it receives a fixed reward of value `healthy_reward`,
    - *forward_reward*: A reward of walking forward which is measured as
    *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
    *dt* is the time between actions and is dependeent on the frame_skip parameter
    (default is 4), where the frametime is 0.002 - making the default
    *dt = 4 * 0.002 = 0.008*. This reward would be positive if the walker walks forward (right) desired.
    - *ctrl_cost*: A cost for penalising the walker if it
    takes actions that are too large. It is measured as
    *`ctrl_cost_weight` * sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is
    a parameter set for the control and has a default value of 0.001

    The total reward returned is ***reward*** *=* *healthy_reward bonus + forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ### Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.

    ### Episode End
    The walker is said to be unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite
    2. The height of the walker is ***not*** in the closed interval specified by `healthy_z_range`
    3. The absolute value of the angle (`observation[1]` if `exclude_current_positions_from_observation=False`, else `observation[2]`) is ***not*** in the closed interval specified by `healthy_angle_range`

    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode ends when any of the following happens:

    1. Truncation: The episode duration reaches a 1000 timesteps
    2. Termination: The walker is unhealthy

    If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

    ### Arguments

    No additional arguments are currently supported in v2 and lower.

    ```
    env = gym.make('Walker2d-v4')
    ```

    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

    ```
    env = gym.make('Walker2d-v4', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default          | Description                                                                                                                                                       |
    | -------------------------------------------- | --------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"walker2d.xml"` | Path to a MuJoCo model                                                                                                                                            |
    | `forward_reward_weight`                      | **float** | `1.0`            | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `1e-3`           | Weight for _ctr_cost_ term (see section on reward)                                                                                                                |
    | `healthy_reward`                             | **float** | `1.0`            | Constant reward given if the ant is "healthy" after timestep                                                                                                      |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the walker is no longer healthy                                                                               |
    | `healthy_z_range`                            | **tuple** | `(0.8, 2)`       | The z-coordinate of the top of the walker must be in this range to be considered healthy                                                                          |
    | `healthy_angle_range`                        | **tuple** | `(-1, 1)`        | The angle must be in this range to be considered healthy                                                                                                          |
    | `reset_noise_scale`                          | **float** | `5e-3`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |


    ### Version History

    * v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
    * v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 125,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )
        
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
        self.ball_pos = np.array([0.0, 0.0, 0.12])  # Ball position
        self.ball_size = 0.12  # Radius of the ball

        # 4. Humanoid Dimensions and Position
        self.humanoid_pos = np.array([0.0, 0.0, 0.0])  # Humanoid starting position
        self.humanoid_size = np.array([1.0, 0.5, 1.8])  # Rough bounding box (for reference)

        # 5. Distance Between Ball and Humanoid
        self.distance_to_ball = np.linalg.norm(self.ball_pos - self.humanoid_pos)

       #Modify the XML to add the ball and goal 
        # 6. Read Base XML
        gym_path = Path(gym.__file__).parent
        xml_file = gym_path / "envs/mujoco/assets/walker2d.xml"
        print(xml_file)
        with open(xml_file, 'r') as f:
            xml_contents = f.read()

        self.time_stamp = time.time()
        # 7. Generate Modular XML
        ball_and_goal = f"""
            <body name="ball" pos="{self.ball_pos[0]} {self.ball_pos[1]} {self.ball_pos[2]}">
                <joint name="ball_free" type="free"/>
                <geom name="ball" type="sphere" size="{self.ball_size}" rgba="1 0 0 1" 
                    mass="0.45" friction="0.8" condim="4" contype="1" conaffinity="1"/>
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


        MujocoEnv.__init__(
            self, self.temp_path, 4, observation_space=observation_space, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]

        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        is_healthy = healthy_z and healthy_angle

        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation
    def compute_reward(
        self,
        walker_pos,
        ball_pos,
        ball_vel,
        goal_pos,
        walker_angular_vel,
        forward_reward,
        proximity_weight=1.0,
        angular_vel_weight=2.0,
        ball_vel_weight=5.0,
        goal_proximity_weight=10.0,
        goal_completion_bonus=100.0
    ):
        """
        Reward function for training the Walker2d to kick the ball into the goal, using hinge angular velocity.
        
        Args:
        - walker_pos: np.array, position of the walker (x, y, z).
        - ball_pos: np.array, position of the ball (x, y, z).
        - ball_vel: np.array, velocity of the ball (x, y, z).
        - goal_pos: np.array, position of the goal center (x, y, z).
        - walker_angular_vel: np.array, angular velocity of the walker joints.
        - forward_reward: float, reward defined for forward movement.
        - proximity_weight: float, weight for the proximity reward.
        - angular_vel_weight: float, weight for the angular velocity reward.
        - ball_vel_weight: float, weight for the ball velocity reward.
        - goal_proximity_weight: float, weight for the goal proximity reward.
        - goal_completion_bonus: float, bonus for getting the ball into the goal.

        Returns:
        - reward: float, total reward for the current step.
        """
        # Distance from walker to the ball
        walker_to_ball_dist = np.linalg.norm(ball_pos - walker_pos)
        ball_to_goal_dist = np.linalg.norm(ball_pos - goal_pos)

        # Ball Proximity Reward: Encourages moving closer to the ball
        ball_proximity_reward = proximity_weight / (1.0 + walker_to_ball_dist)

        # Angular Velocity Reward: Encourage higher angular velocities near the ball
        angular_vel_reward = angular_vel_weight * np.sum(
            walker_angular_vel
        ) / (1.0 + walker_to_ball_dist)

        # Kicking Reward: Encourage increasing ball velocity
        kicking_reward = ball_vel_weight * np.linalg.norm(ball_vel)

        # Goal Proximity Reward: Encourage reducing ball distance to the goal
        goal_proximity_reward = goal_proximity_weight / (1.0 + ball_to_goal_dist)

        # Goal Completion Bonus: Large reward for getting the ball in the goal
        goal_completion_reward = 0.0
        if ball_to_goal_dist < 0.5:  # Assume ball is "in goal" if within 0.5 units
            goal_completion_reward = goal_completion_bonus

        # Combine rewards
        reward = (
            forward_reward
            + ball_proximity_reward
            + angular_vel_reward
            + kicking_reward
            + goal_proximity_reward
            + goal_completion_reward
        )
        return reward


    def step(self, action):
        x_position_before = self.data.qpos[0]
        ball_id = None
        for i in range(self.model.nbody):
            body_name = self.model.names[self.model.name_bodyadr[i]:].split(b'\x00', 1)[0].decode('utf-8')
            if body_name == 'ball':
                ball_id = i
                break

        if ball_id is None:
            raise ValueError("Body 'ball' not found in the model.")
        self.ball_pos = self.data.xpos[ball_id]
        self.do_simulation(action, self.frame_skip)
        # Extract positions and velocities
        walker_pos = self.data.qpos[:3].copy()
        ball_pos = self.ball_pos
        ball_vel = self.data.qvel[9:12].copy()  # Example indices for ball velocity
        walker_angular_vel = self.data.qvel[3:9].copy()  # Example indices for angular velocity
        goal_pos = np.array([17.5, 0.0, 0.0])  # Goal position (example)

        # Compute forward reward (already defined)
        x_position_after = self.data.qpos[0]
        x_position_before = self.data.qpos[0] - self.dt * self.data.qvel[0]  # Approx previous x-position
        forward_reward = self._forward_reward_weight * (x_position_after - x_position_before) / self.dt

        # Compute custom reward
        reward = self.compute_reward(
            walker_pos=walker_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            goal_pos=goal_pos,
            walker_angular_vel=walker_angular_vel,
            forward_reward=forward_reward
        )
        def euclidean_distance(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.linalg.norm(a - b)
        
        observation = self._get_obs()
        velocity = abs(np.average(self.data.qvel[9:12].copy()))
        moving = velocity >= 1e-5
        if moving:
            self.time_stamp = time.time()
        terminated = euclidean_distance(self.ball_pos, x_position_after) >= 0.75 and not moving

        if time.time() - self.time_stamp >= 4:
            terminated = True
        info = {
            "x_position": x_position_after,
            "x_velocity": self.data.qvel[0],
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

if __name__ == "__main__":
    # Create and register the environment
    env = Walker2dEnv(render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    
    try:
        for _ in range(10000):
            # Random action
            action = env.action_space.sample()
            
            # Step the environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("terminating!")
                observation, info = env.reset()
                env.time_stamp = time.time()
    finally:
        env.close()