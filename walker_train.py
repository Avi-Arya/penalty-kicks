import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import sys
from walker_demo import Walker2dEnv
from soccer_env import HumanoidSoccerEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv



print(f"Current working directory: {sys.path[0]}")
class ProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress and debugging
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log episode-level information
        # for done in self.locals['dones']:
        #     if done:
        #         # Capture episode reward and length
        #         episode_reward = np.sum(self.locals['rewards'])
        #         episode_length = len(self.locals['rewards'])
                
        #         self.episode_rewards.append(episode_reward)
        #         self.episode_lengths.append(episode_length)
                
        #         # Periodic logging
        #         if len(self.episode_rewards) % 10000 == 0:
        #             print(f"Last 1000 Episodes - Avg Reward: {np.mean(self.episode_rewards[-10000:]):.2f}, "
        #                   f"Avg Length: {np.mean(self.episode_lengths[-10000:]):.2f}")
        
        return True


class HumanoidSoccerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Enhanced feature extractor with more strategic architecture
    Focuses on dimensional reduction and feature learning
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Strategic feature extraction
        self.feature_extractor = nn.Sequential(
            # Initial layer with dimensional reduction
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),  # Modern activation function
            nn.Dropout(0.2),
            
            # Feature learning layers
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.15),
            
            # Final feature representation
            nn.Linear(192, features_dim),
            nn.LayerNorm(features_dim)
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Apply feature extraction
        features = self.feature_extractor(observations)
        return features

class CustomHumanoidSoccerPolicy(ActorCriticPolicy):
    """
    Enhanced Actor-Critic Policy with modular and robust architecture
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        net_arch = None,
        activation_fn = nn.GELU,  # Modern activation function
        *args,
        **kwargs
    ):
        # More refined network architecture
        if net_arch is None:
            net_arch = dict(
                pi=[256, 192, 128],  # Policy network
                vf=[256, 192, 128]   # Value function network
            )
        
        # Use custom features extractor
        kwargs['features_extractor_class'] = HumanoidSoccerFeaturesExtractor
        kwargs['features_extractor_kwargs'] = dict(features_dim=128)
        
        # Initialize the policy
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """
        Enhanced MLP extractor with advanced regularization
        """
        super()._build_mlp_extractor()
        
        # Find the last linear layer to get out_features
        policy_last_linear = [m for m in self.mlp_extractor.policy_net.modules() if isinstance(m, nn.Linear)][-1]
        value_last_linear = [m for m in self.mlp_extractor.value_net.modules() if isinstance(m, nn.Linear)][-1]
        
        # Reconstruct the policy and value networks with additional regularization
        self.mlp_extractor.policy_net = nn.Sequential(
            self.mlp_extractor.policy_net,
            nn.Dropout(0.2),
            nn.LayerNorm(policy_last_linear.out_features)
        )
        
        self.mlp_extractor.value_net = nn.Sequential(
            self.mlp_extractor.value_net,
            nn.Dropout(0.2),
            nn.LayerNorm(value_last_linear.out_features)
        )


def create_env(render_mode="rgb_array"):
    """
    Environment creation with potential curriculum learning
    """
    env = HumanoidEnv()
    return env

def train():
    # Configuration
    total_timesteps = 500000  # Increased training duration
    n_envs = 32  # Parallel environments for better data collection
    
    # Create vectorized environments
    env = make_vec_env(create_env, n_envs=n_envs)
    
    # Create evaluation environment
    eval_env = make_vec_env(create_env, n_envs=1)

    # Callbacks
    progress_callback = ProgressCallback()
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/',
        log_path='./logs/', 
        eval_freq=10000,
        deterministic=True, 
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000, 
        save_path='./models/',
        name_prefix='ppo_humanoid_soccer'
    )

    # PPO Hyperparameters (Tuned for complex continuous control)
    model = PPO(
        policy=CustomHumanoidSoccerPolicy, 
        env=env, 
        verbose=1, 
        tensorboard_log="./humanoid_final_tensorboard/",
        
        learning_rate=1e-3,  # Try a lower learning rate
        n_steps=4096,        # Increase batch size
        batch_size=128,      # Larger batch
        n_epochs=5,          # Fewer optimization epochs
        gamma=0.99,          
        gae_lambda=0.9,      # Adjust advantage estimation
        clip_range=0.1,      # Tighter clipping
        ent_coef=0.005,      # Reduce entropy coefficient
        vf_coef=0.5,
        max_grad_norm=1.0    # Looser gradient clipping
    )

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[progress_callback, eval_callback, checkpoint_callback]
    )

    # Save final model
    model.save("ppo_modifed_final")
    print("Model saved as ppo_modifed_final.zip")

    # Close environments
    env.close()
    eval_env.close()

# Visualization function (optional)
def visualize_policy(env, model, episodes=5):
    """
    Visualize the trained policy
    """
    for episode in range(episodes):
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        print(f"Episode {episode + 1}/{episodes}")
        
        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward[0]
            step += 1
        
        print(f"Episode {episode + 1} - Total Reward: {total_reward}, Steps: {step}")

# Entry point
if __name__ == "__main__":
    train()