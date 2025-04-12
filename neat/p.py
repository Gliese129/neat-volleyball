from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class HyperParams:
    """Hyperparameters for NEAT algorithm."""
    population_size: int = 100
    max_generations: int = 1000

    # possibility of mutation
    crossover_rate: float = 0.5
    mutation_rate: float = 0.3
    enable_rate: float = 0.2
    mutate_weight_rate: float = 0.7
    mutate_activation_rate: float = 0.3

    elite_ratio: float = 0.2

    specie_threshold: float = 0.2
    speciate_gene_coefficient: float = 0.5
    speciate_weight_coefficient: float = 0.5


activation_functions = [
    lambda x: 1 / (1 + jnp.exp(-x)), # sigmoid
    lambda x: jnp.tanh(x), # tanh
    lambda x: jnp.maximum(0, x), # ReLU
    lambda x: jnp.maximum(-1, x), # Leaky ReLU
]

