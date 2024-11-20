# ppo_agent.py
from stable_baselines3 import PPO

class PPOAgent:
    def __init__(self, env, policy="MlpPolicy"):
        self.env = env
        self.model = PPO(policy, env, verbose=0)

    def train(self, timesteps):
        """Train the agent using PPO for the specified number of timesteps."""
        self.model.learn(total_timesteps=timesteps)

    def evaluate(self, num_episodes=5):
        """Evaluate the agent's performance over multiple episodes."""
        total_reward = 0
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
        return total_reward / num_episodes

    def get_weights(self):
        """Retrieve the model weights."""
        return self.model.policy.state_dict()

    def set_weights(self, weights):
        """Set the model weights."""
        self.model.policy.load_state_dict(weights)
