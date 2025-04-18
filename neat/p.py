from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class HyperParams:
    """Hyperparameters for NEAT algorithm."""
    population_size: int = 100
    max_generations: int = 1000

    input_size: int = 8
    output_size: int = 3
    init_activation: int = 0 # sigmoid as default activation function
    init_enabled_prob: float = 0.5 # probability of a connection being enabled

    use_speciation: bool = True
    min_speciate_threshold: float = 2.0
    speciate_threshold_movement: float = 0.25
    ideal_species: int = 4
    stagnation_limit: int = 4 # number of generations without improvement

    use_dominated_sorting_prob: float = 0.5 # probability of using dominated sorting

    # possibility of mutation
    crossover_rate: float = 0.5
    mutation_rate: float = 0.3
    enable_rate: float = 0.2
    mutate_weight_rate: float = 0.7
    mutate_activation_rate: float = 0.3
    mutate_add_node_rate: float = 0.3
    mutate_add_edge_rate: float = 0.3

    # possibility of crossover
    elite_ratio: float = 0.2
    cull_ratio: float = 0.2
    rolling_num: int = 5 # number of parents to select for crossover

    specie_threshold: float = 0.2
    speciate_gene_coefficient: float = 0.5
    speciate_weight_coefficient: float = 0.5

    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.4


activation_functions = [
    lambda x: 1 / (1 + jnp.exp(-x)), # sigmoid
    lambda x: jnp.tanh(x), # tanh
    lambda x: jnp.maximum(0, x), # ReLU
    lambda x: jnp.maximum(-1, x), # Leaky ReLU
]

