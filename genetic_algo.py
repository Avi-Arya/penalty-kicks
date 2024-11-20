# genetic_algorithm.py
import torch
import numpy as np

def mutate(weights, mutation_rate=0.1):
    """Apply random mutations to the weights."""
    mutated_weights = {}
    for key, value in weights.items():
        if torch.is_tensor(value):
            noise = torch.randn_like(value) * mutation_rate
            mutated_weights[key] = value + noise
        else:
            mutated_weights[key] = value
    return mutated_weights

def crossover(parent1_weights, parent2_weights):
    """Combine weights from two parents to create an offspring."""
    child_weights = {}
    for key in parent1_weights.keys():
        if torch.is_tensor(parent1_weights[key]):
            mask = torch.rand_like(parent1_weights[key]) > 0.5
            child_weights[key] = torch.where(mask, parent1_weights[key], parent2_weights[key])
        else:
            child_weights[key] = parent1_weights[key]
    return child_weights

def select_top_agents(agents, fitnesses, top_percent=0.2):
    """Select the top-performing agents based on fitness."""
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
    num_selected = int(len(agents) * top_percent)
    return [agents[i] for i in sorted_indices[:num_selected]]
