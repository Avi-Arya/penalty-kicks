import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from environ import HumanoidSoccerEnv3D  # Import your environment

# Custom Policy Network (optional, extend for PyTorch customization)
class CustomPolicy(PPO.policy_class):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU,
            **kwargs
        )
        self.mlp_extractor = MlpExtractor(
            observation_space.shape[0],
            policy_network_arch=[64, 64],
            value_network_arch=[64, 64]
        )

# Main function for training
if __name__ == "__module__":
    pass


if __name__ == "__main__":
    # Instantiate the environment
    env = HumanoidSoccerEnv3D(render_mode="human", humanoid_urdf_path="humanoid.urdf")
    
    # Wrap the environment for vectorization
    vec_env = make_vec_env(lambda: env, n_envs=4)

    # Define a custom checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save the model every 10,000 steps
        save_path="./checkpoints/",
        name_prefix="humanoid_ppo"
    )

    # Create the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        tensorboard_log="./tensorboard_logs/"
    )

    # Train the model
    model.learn(
        total_timesteps=500000,  # Train for 500,000 timesteps
        callback=checkpoint_callback
    )

    # Save the final model
    model.save("ppo_humanoid_soccer")

    # Evaluate the trained model
    env = HumanoidSoccerEnv3D(render_mode="human", humanoid_urdf_path="humanoid.urdf")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        if done:
            obs, _ = env.reset()

    env.close()
