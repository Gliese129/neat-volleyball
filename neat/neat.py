import copy
import functools
from typing import List

import jax.numpy as jnp
import jax.random as random

from . import Individual
from .p import HyperParams
from ._speciate import Specie, speciate, assign_species, assign_specie_offspring
from .nsga2 import nsga_sort


class Neat:
    """

    Attributes:
        population: List of individuals in the population.
        species: List of species in the population.
        generation: Current generation number.
        innovation_record: Record of innovations (connections and nodes) in the population.
            - [0, :] = innovation id
            - [1, :] = source node id
            - [2, :] = target node id
        species_threshold: Ideal threshold for speciation.
    """
    speciate: callable
    assign_species: callable
    assign_specie_offspring: callable

    p: HyperParams
    population: List[Individual]
    species: List[Specie]
    generation: int
    innovation_record: jnp.ndarray
    species_threshold: float

    def __init__(self, p: HyperParams):
        self.p = p
        self.population = []
        self.species = []
        self.generation = 0
        self.innovation_record = jnp.zeros((3, 0), dtype=jnp.int32)
        self.species_threshold = p.min_speciate_threshold

        self.speciate = functools.partial(speciate, self)
        self.assign_species = functools.partial(assign_species, self)
        self.assign_specie_offspring = functools.partial(assign_specie_offspring, self)

    def tell(self, rewards: jnp.ndarray):
        """
        Update the fitness of the population based on the rewards received.
        """
        for i, individual in enumerate(self.population):
            individual.fitness = rewards[i].item()
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

    def ask(self):
        """
        Generate a new population of individuals based on the current population.
        """
        key = random.PRNGKey(self.generation)
        if len(self.population) == 0:
            self.init_pop(key)
        else:
            self.rank_individuals(key=key)
            self.speciate()
            self.evolve()
        return self.population

    def rank_individuals(self, key: jnp.ndarray):
        """
        Rank the individuals in the population based on their fitness.
        """
        # Sort the population based on fitness
        fitness = jnp.array([individual.fitness for individual in self.population])
        connections = jnp.array([individual.conn_cnt for individual in self.population])
        connections = connections.at[connections == 0].set(1) # Avoid division by 0

        fitness = -fitness # Larger fitness is better
        attrs = jnp.stack([fitness, 1 / connections], axis=1) # Fewer connections is better

        key, subkey = random.split(key)
        if random.uniform(subkey) < self.p.use_dominated_sorting_prob:
            # Use dominated sorting
            rank = nsga_sort(attrs)
        else:
            # Use fitness sorting
            rank = jnp.argsort(fitness)
        # Assign ranks to individuals
        for i, individual in enumerate(self.population):
            individual.rank = rank[i].item()

    def evolve(self):
        """ Evolves new population from existing species.
            Wrapper which calls 'recombine' on every species and combines all offspring into a new population. When speciation is not used, the entire population is treated as a single species.
        """
        new_pop = []
        self.generation += 1
        for i in range(len(self.species)):
            children, self.innovation_record = self.recombine(self.species[i], self.innovation_record, self.generation,
                                                              key=random.PRNGKey(i))
            new_pop.append(children)
        self.population = [child for specie in new_pop for child in specie]

    def recombine(self, specie: Specie, innovation_record: jnp.ndarray, generation: int, key: jnp.ndarray):
        """ Creates next generation of child solutions from a species
            Procedure:
            - Sort all individuals by rank
            - Eliminate lower percentage of individuals from breeding pool
            - Pass upper percentage of individuals to child population unchanged
            - Select parents by tournament selection
            - Produce new population through crossover and mutation
        """
        n_offspring = int(specie.offspring_num)
        population = specie.members
        children = []

        # Sort the population by rank
        population.sort(key=lambda x: x.rank)

        # Cull the population
        n_cull = int(len(population) * self.p.cull_ratio)
        if n_cull > 0:
            population = population[:-n_cull]

        # Select the elite individuals
        n_elite = int(len(population) * self.p.elite_ratio)
        children.extend(population[:n_elite])
        n_offspring -= n_elite

        # Select parents for crossover
        parent_a = random.choice(key, len(population), shape=(n_offspring,), replace=True)
        parent_b = random.choice(key, len(population), shape=(n_offspring,), replace=True)
        parents = jnp.where(parent_a[:, None] < parent_b[:, None],
                            jnp.stack([parent_a, parent_b], axis=1),
                            jnp.stack([parent_b, parent_a], axis=1))

        # Create children
        for i in range(n_offspring):
            key, subkey = random.split(key)
            if random.uniform(subkey) < self.p.crossover_rate:
                # Perform crossover
                child, innovation_record = population[parents[i, 0]].create_child(self.p, innovation_record, generation, other=population[parents[i, 1]])
            else:
                child, innovation_record = population[parents[i, 0]].create_child(self.p, innovation_record, generation)
            children.append(child)
        return children, innovation_record


    def init_pop(self, key: jnp.ndarray):
        p = self.p
        # Initialize the population with random individuals
        node_ids = jnp.arange(p.input_size + p.output_size + 1)
        nodes = jnp.zeros((3, p.input_size + p.output_size + 1), dtype=jnp.int32)
        nodes = nodes.at[0, :].set(node_ids)
        # node types
        nodes = nodes.at[1, 0].set(4) \
                     .at[1, 1:(p.input_size + 1)].set(1) \
                     .at[1, (p.input_size + 1):].set(2)
        # activation functions
        nodes = nodes.at[2, :].set(p.init_activation)

        # connections
        n_connections = (p.input_size + 1) * p.output_size # input+bias to output
        inputs = jnp.arange(p.input_size + 1)
        outputs = jnp.arange(p.input_size + 1, p.input_size + 1 + p.output_size)

        connections = jnp.zeros((5, n_connections), dtype=jnp.float32)
        connections = connections.at[0, :].set(jnp.arange(n_connections)) # innovation id
        connections = connections.at[1, :].set(jnp.tile(inputs, p.output_size)) # source node
        connections = connections.at[2, :].set(jnp.repeat(outputs, p.input_size + 1)) # target node
        connections = connections.at[3, :].set(jnp.nan) # weight
        connections = connections.at[4, :].set(1) # enabled

        # Create population of individuals with varied weights
        population = []
        for i in range(p.population_size):
            key, subkey = random.split(key)
            weights = random.uniform(subkey, shape=(n_connections,), minval=-1, maxval=1)
            key, subkey = random.split(key)
            enabled = random.uniform(subkey, shape=(n_connections,), minval=0, maxval=1) < p.init_enabled_prob
            enabled = enabled.astype(jnp.int32)
            connections_ = connections.copy()
            connections_ = connections_.at[3, :].set(weights)
            connections_ = connections_.at[4, :].set(enabled)
            new_individual = Individual(nodes.copy(), connections_.copy())
            new_individual.generation = 0
            population.append(new_individual)

        # Create innovation record
        innovation_record = jnp.zeros((3, n_connections), dtype=jnp.int32)
        innovation_record = innovation_record.at[0:3, :].set(connections[0:3, :])

        self.population = population
        self.innovation_record = innovation_record

    def to_json(self):
        """
        Convert the NEAT object to a JSON serializable format.
        """
        return {
            "population": [ind.to_json() for ind in self.population],
            "species": [specie.to_json() for specie in self.species],
            "generation": self.generation,
            "innovation_record": self.innovation_record.tolist(),
            "species_threshold": self.species_threshold,
            "p": self.p.to_json(),
            "_id_counter": Individual._id_counter,
        }

    @classmethod
    def from_json(cls, json_str):
        """
        Create a NEAT object from a JSON serializable format.
        """
        data = json_str
        p = HyperParams.from_json(data["p"])
        neat = cls(p)
        neat.generation = data["generation"]
        neat.species_threshold = data["species_threshold"]
        neat.innovation_record = jnp.array(data["innovation_record"])
        neat.population = [Individual.from_json(ind) for ind in data["population"]]
        neat.species = [Specie.from_json(specie) for specie in data["species"]]

        Individual._id_counter = data["_id_counter"]
        return neat