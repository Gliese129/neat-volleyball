from dataclasses import dataclass
import jax.numpy as jnp

@dataclass
class HyperParams:
    # ========== Evolution scale ==========
    population_size: int = 150          # individuals per generation
    max_generations: int = 400          # training budget

    # ========== Network topology ==========
    input_size: int = 12                 #
    output_size: int = 3                # move‑left, move‑right, jump
    init_activation: int = 2            # 0:Sigmoid,1:Tanh,2:ReLU,3:LeakyReLU
    init_enabled_prob: float = 0.80     # initial probability a connection is enabled

    # ========== Speciation ==========
    use_speciation: bool = True
    min_speciate_threshold: float = 1.5 # lower bound on compatibility threshold
    speciate_threshold_movement: float = 0.20  # step size to approach ideal_species
    ideal_species: int = 5              # target number of species
    stagnation_limit: int = 6           # generations without improvement before “stagnant”

    # ========== Ranking method ==========
    use_dominated_sorting_prob: float = 0.30   # chance to switch to NSGA‑II sorting

    # ========== Mutation / Crossover probabilities ==========
    crossover_rate: float = 0.75
    mutation_rate: float  = 0.25       # not used directly here, but handy for wrappers
    enable_rate: float    = 0.20
    mutate_weight_rate: float = 0.80
    mutate_activation_rate: float = 0.10
    mutate_add_node_rate: float = 0.20
    mutate_add_edge_rate: float = 0.40

    # ========== Elitism & Cull ==========
    elite_ratio: float = 0.10          # top fraction copied unchanged
    cull_ratio:  float = 0.40          # worst fraction removed before breeding
    rolling_num: int = 5               # tournament size when selecting parents

    # ========== Compatibility coefficients (distance = c1*E+c2*D+c3*W̄) ==========
    c1: float = 1.0                    # excess genes
    c2: float = 1.0                    # disjoint genes
    c3: float = 0.4                    # average weight difference

    def to_json(self):
        """
        Convert the HyperParams object to a JSON serializable dictionary.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

    @classmethod
    def from_json(cls, json_data):
        """
        Create a HyperParams object from a JSON serializable dictionary.
        """
        return cls(**json_data)


# ---------- Activation‑function lookup table ----------
activation_functions = [
    lambda x: 1 / (1 + jnp.exp(-x)),          # 0: Sigmoid
    lambda x: jnp.tanh(x),                    # 1: Tanh
    lambda x: jnp.maximum(0, x),              # 2: ReLU
    lambda x: jnp.where(x > 0, x, 0.01 * x)   # 3: Leaky‑ReLU (slope = 0.01)
]
