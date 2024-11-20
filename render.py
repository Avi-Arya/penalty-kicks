import mujoco_py  # or gymnasium if using gymnasium
import numpy as np
import time
from stable_baselines3 import PPO
import cv2

# Load the MuJoCo model
model = PPO.load("humanoid_soccer_model.zip")

# Create a MuJoCo environment
mj_model = mujoco_py.load_model_from_path("humanoid.xml")
sim = mujoco_py.MjSim(mj_model)

# Create the viewer for rendering
viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=0)
viewer.cam.type = mujoco_py.generated.const.CAMERA_FREE
viewer.cam.fixedcamid = -1
viewer.cam.distance = 5  # Set the camera distance from the humanoid
viewer.cam.azimuth = 90  # Set the horizontal angle of the camera
viewer.cam.elevation = -10  # Set the vertical angle of the camera
viewer.cam.lookat[0] = sim.data.get_body_xpos("torso")[0]  # Focus on the torso or main body
viewer.cam.lookat[1] = sim.data.get_body_xpos("torso")[1]
viewer.cam.lookat[2] = sim.data.get_body_xpos("torso")[2]

# Load the trained policy from Stable-Baselines3
trained_policy = PPO.load("humanoid_soccer_model.zip")

# Define the video writer using OpenCV
video_path = "/content/humanoid_simulation.mp4"
video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

# Simulate the environment using the trained policy
def policy_kick_simulation(steps=1000):
    # Get the initial observation using the same method as in your environment
    obs = np.concatenate([sim.data.qpos.flat, sim.data.qvel.flat, sim.data.get_body_xpos("ball")])
    for i in range(steps):
        # Predict the action using the trained policy
        action, _states = trained_policy.predict(obs, deterministic=True)

        # Apply the action to the humanoid's actuators
        sim.data.ctrl[:] = action

        # Step the simulation forward
        sim.step()
        print(sim.data.qpos)

        # Get the observation for the next step
        obs = np.concatenate([sim.data.qpos.flat, sim.data.qvel.flat, sim.data.get_body_xpos("ball")])

        # Capture the frame as a NumPy array (640x480 is the frame size)
        img = viewer.read_pixels(640, 480, depth=False)
        img = np.flipud(img)  # Flip image vertically because MuJoCo renders it upside-down

        # Write the frame to the video file
        video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Slow down the simulation for visualization
        time.sleep(0.01)  # Adjust the speed for visualization

    video_writer.release()  # Release the video writer when done

# Run the simulation with the trained policy
policy_kick_simulation(steps=100)