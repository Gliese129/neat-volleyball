from dataclasses import dataclass

from . import Individual
# from .neat import Neat # TODO: remove this when running to avoid circular import
import jax.numpy as jnp


@dataclass
class Specie:
    """
    Class representing a specie in NEAT (NeuroEvolution of Augmenting Topologies).
    """
    represent: Individual
    members: list[Individual]
    best_individual: Individual
    best_fitness: float
    last_improvement_elapsed: int
    offspring_num: int

    def __init__(self, represent: Individual):
        self.represent = represent
        self.members = []
        self.best_individual = represent
        self.best_fitness = represent.fitness
        self.last_improvement_elapsed = 0
        self.offspring_num = 0

    def to_json(self):
        """
        Convert the Specie object to a JSON-compatible dictionary.
        """
        return {
            'represent': self.represent.to_json(),
            'members': [member.to_json() for member in self.members],
            'best_individual': self.best_individual.to_json(),
            'best_fitness': self.best_fitness,
            'last_improvement_elapsed': self.last_improvement_elapsed,
            'offspring_num': self.offspring_num,
        }

    @classmethod
    def from_json(cls, json_data):
        """
        Create a Specie object from a JSON-compatible dictionary.
        """
        represent = Individual.from_json(json_data['represent'])
        members = [Individual.from_json(member) for member in json_data['members']]
        best_individual = Individual.from_json(json_data['best_individual'])
        best_fitness = json_data['best_fitness']
        last_improvement_elapsed = json_data['last_improvement_elapsed']
        offspring_num = json_data['offspring_num']
        specie = cls(represent)
        specie.members = members
        specie.best_individual = best_individual
        specie.best_fitness = best_fitness
        specie.last_improvement_elapsed = last_improvement_elapsed
        specie.offspring_num = offspring_num
        return specie


def speciate(self: 'Neat'):
    p = self.p
    species = self.species

    if p.use_speciation:
        if len(species) > p.ideal_species:
            self.species_threshold += p.speciate_threshold_movement
        elif len(species) < p.ideal_species:
            self.species_threshold -= p.speciate_threshold_movement
        if self.species_threshold < p.min_speciate_threshold:
            self.species_threshold = p.min_speciate_threshold

        self.assign_species()
        self.assign_specie_offspring()


def assign_species(self: 'Neat'):
    """
    Assigns individuals to species based on their genetic similarity.
    """
    species = self.species
    if len(species) == 0:
        species = [Specie(self.population[0])]
    else:
        # clear all members
        for specie in species:
            specie.members = []
    # assign individuals to species
    for individual in self.population:
        for idx, specie in enumerate(species):
            if specie.represent.distance(individual, p=self.p) < self.species_threshold:
                specie.members.append(individual)
                individual.specie = idx
                break
        else:
            # if not found, create new specie
            species.append(Specie(individual))
            individual.specie = len(species) - 1
    for specie in species:
        # remove empty species
        if len(specie.members) == 0:
            species.remove(specie)
        # update representative
        specie.represent = specie.members[0]
    self.species = species

def assign_specie_offspring(self: 'Neat'):
    """
    Assigns offspring to species based on their genetic similarity.
    """
    assert len(self.species) > 0, "Species list is empty. Cannot assign offspring."
    if len(self.species) == 1:
        self.species[0].offspring_num = self.p.population_size
        return
    # sort individuals by fitness(use average ranking)
    pop_fitness = jnp.array([ind.fitness for ind in self.population])
    rank = jnp.argsort(pop_fitness).astype(jnp.float32)
    idx = 0
    while idx < len(rank):
        cnt = jnp.sum(pop_fitness == pop_fitness[idx])
        rank = rank.at[idx:idx + cnt].set(jnp.mean(rank[idx:idx + cnt]))
        idx += cnt
    score = 1 / rank  # using exponential score

    species_id = jnp.array([individual.specie for individual in self.population])
    species_cnt = len(self.species)
    species_fitness = jnp.zeros((species_cnt,), dtype=jnp.float32) # average fitness of species(using score)
    for i in range(species_cnt):
        if jnp.sum(species_id == i) == 0: # no individual in this species
            continue
        species_fitness = species_fitness.at[i].set(jnp.mean(score[species_id == i]))
        best_fitness = jnp.max(pop_fitness[species_id == i])
        best_idx = jnp.argmax(pop_fitness[species_id == i])
        # check if this species improved
        if self.species[i].best_fitness < best_fitness:
            self.species[i].last_improvement_elapsed = 0
            self.species[i].best_fitness = best_fitness
            self.species[i].best_individual = self.population[best_idx]
        else:
            self.species[i].last_improvement_elapsed += 1

        if self.species[i].last_improvement_elapsed > self.p.stagnation_limit:
            species_fitness = species_fitness.at[i].set(0.0)

    # assign offspring number
    if jnp.sum(species_fitness) == 0:
        # all species are stagnated, assign equally(could this happen?)
        species_fitness = jnp.ones((species_cnt,), dtype=jnp.float32)
        print("WARN: All species are stagnated, assign equally.")

    offspring = _best_int_split(species_fitness, self.p.population_size)
    # assign offspring number to species
    for i in range(species_cnt):
        self.species[i].offspring_num = offspring[i]


def _best_int_split(ratio: jnp.ndarray, total: int):
    """
    Split the total into parts based on the ratio.
    :param ratio: The ratio to split the total.
    :param total: The total to split.
    :return: The split parts.
    """
    # normalize the ratio
    ratio = ratio / jnp.sum(ratio)
    float_split = ratio * total
    int_split = jnp.floor(float_split).astype(jnp.int32)
    remainder = total - jnp.sum(int_split)
    # distribute the remainder
    deserving = jnp.argsort(float_split - jnp.floor(float_split))
    for i in range(remainder):
        int_split = int_split.at[deserving[i]].set(int_split[deserving[i]] + 1)
    return int_split



