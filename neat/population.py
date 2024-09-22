import random
from typing import Optional
import jax.numpy as jnp

from neat.genome import Genome
from neat.species import Species


class Population:
    max_size: int
    species: list[Species]
    organisms: list[Genome]
    best: Optional[Genome]
    steps: int = 0
    fittest_func: any

    def __init__(self, init_organisms: list[Genome], max_size: int, fittness_func):
        self.max_size = max_size
        self.organisms = init_organisms
        self.species = []
        self.fittest_func = fittness_func

        self.set_fitness()

    def speciate(self):
        for organism in self.organisms:
            for specie in self.species:
                if specie.can_have(organism):
                    specie.genomes.append(organism)
                    break
            else:
                self.species.append(Species(organism))

    def set_fitness(self):
        for organism in self.organisms:
            organism.fitness = self.fittest_func(organism, self)
        self.organisms.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.organisms[0]

    def step(self):
        from .superparams import mutation_rate, crossover_rate, species_best_size
        self.steps += 1
        self.speciate()
        # generate the next generation
        new_organisms = []
        for specie in self.species:
            specie.set_adjusted_fitness()
            # eliminate the worst organisms
            specie.genomes.sort(key=lambda x: x.fitness, reverse=True)
            specie.genomes = specie.genomes[:species_best_size]
            # elitism
            new_organisms.append(specie.best)
            # crossover
            for _ in range(species_best_size):
                weights = [genome.adjusted_fitness for genome in specie.genomes]
                sum_ = sum(weights)
                weights = [w / sum_ for w in weights]
                if random.random() < crossover_rate:
                    parent1, parent2 = random.choices(specie.genomes, weights= weights, k=2)
                    try:
                        child = parent1 * parent2
                        new_organisms.append(child)
                    except Exception as e:
                        print(e)
                if random.random() < mutation_rate:
                    organism = random.choices(specie.genomes, weights= weights, k=1)[0]
                    new_organisms.append(organism.mutate())
        self.organisms = new_organisms
        self.set_fitness()
        self.species = []
        self.organisms = self.organisms[:self.max_size]




