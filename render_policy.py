import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from soccer_env import HumanoidSoccerEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv


def render_trained_agent(trained_model_path, episodes=10):
    """
    Render the trained agent playing the environment.
    """
    def create_env():
        return HumanoidEnv(render_mode="human")  # Use "human" for rendering

    # Create the environment
    env = DummyVecEnv([create_env])

    # Load the trained model
    model = PPO.load(trained_model_path)

    # Check observation space consistency
    assert env.observation_space.shape[0] == model.policy.observation_space.shape[0], (
        f"Environment observation space {env.observation_space.shape[0]} "
        f"does not match model's expected observation space {model.policy.observation_space.shape[0]}"
    )

    for episode in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        print(f"Episode {episode + 1}/{episodes}")
        while not done and step < 1000:  # Limit steps to avoid infinite loops
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, _ = env.step(action)
            step += 1

    env.close()

if __name__ == "__main__":
    # for i in range(200000, 2000001, 200000):
    #     trained_model_path = f"./models/humanoid/ppo_humanoid_{i}_steps.zip"
    #     render_trained_agent(trained_model_path)

    trained_model_path = "./models/humanoid/dynamic_decay_ppo/ppo_humanoid_900000_steps"
    render_trained_agent(trained_model_path)
