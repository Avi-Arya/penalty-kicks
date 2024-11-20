# train.py
import numpy as np
from soccer_env import HumanoidSoccerEnv
from ppo_agent import PPOAgent
from genetic_algo import mutate, crossover, select_top_agents

if __name__ == "__main__":
    # Initialize environment
    env = HumanoidSoccerEnv(render_mode="human")

    # Hyperparameters
    population_size = 10
    generations = 50
    ppo_timesteps = 10_000
    mutation_rate = 0.1
    top_percent = 0.2

    # Initialize population of PPO agents
    population = [PPOAgent(env) for _ in range(population_size)]

    for generation in range(generations):
        print(f"Generation {generation + 1}")

        # Step 1: Train each agent with PPO
        for agent in population:
            agent.train(timesteps=ppo_timesteps)

        # Step 2: Evaluate fitness of each agent
        fitnesses = [agent.evaluate() for agent in population]
        print(f"Fitnesses: {fitnesses}")

        # Step 3: Select the top-performing agents
        top_agents = select_top_agents(population, fitnesses, top_percent)

        # Step 4: Create the next generation
        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = np.random.choice(top_agents, 2, replace=False)
            child_weights = crossover(parent1.get_weights(), parent2.get_weights())
            child_weights = mutate(child_weights, mutation_rate)

            child_agent = PPOAgent(env)
            child_agent.set_weights(child_weights)
            next_population.append(child_agent)

        # Replace the old population with the new one
        population = next_population

    print("Training complete.")
