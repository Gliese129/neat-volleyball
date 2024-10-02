import json
import multiprocessing as mp
import os
import random
from copy import deepcopy
from typing import Optional, List
import jax.numpy as jnp

from .genome import Genome
from .global_state import InnovationNumber
from .recorder import Recorder
from .species import Species
from .superparams import checkpoint_path, specie_best_size


class Population:
    max_size: int
    species: List[Species]
    organisms: List[Genome]
    best: Optional[Genome]
    steps: int = 0
    fitness_func: any

    def __init__(self, init_organisms: List[Genome], max_size: int, fittness_func):
        self.max_size = max_size
        self.organisms = init_organisms
        for idx, organism in enumerate(self.organisms):
            organism.generation = 0
            organism.idx = idx

        self.species = []
        self.fitness_func = fittness_func

        self.pool = mp.Pool(mp.cpu_count())


    def speciate(self):
        self.species = []
        for organism in self.organisms:
            for specie in self.species:
                if specie.can_have(organism):
                    specie.organisms.append(organism)
                    break
            else:
                self.species.append(Species(organism))

    @staticmethod
    def compute_fitness(args):
        organism, all_organisms, specie_organisms, fitness_func, current_step = args
        fitness = fitness_func(organism, all_organisms, specie_organisms, current_step)
        return organism, fitness

    def set_fitness(self):
        train_set = []
        idx_map = {}
        for specie in self.species:
            for organism in specie.organisms:
                train_set.append((organism, self.organisms, specie.organisms, self.fitness_func, self.steps))
                idx_map[organism.genome_id_] = organism


        if len(train_set) <= mp.cpu_count():
            results = [self.compute_fitness(train) for train in train_set]
        else:
            results = self.pool.map(self.compute_fitness, train_set, chunksize=8)

        for organism, fitness in results:
            idx_map[organism.genome_id_].fitness = fitness

        self.organisms.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.organisms[0]

    def close(self):
        self.pool.close()
        self.pool.join()

    def step(self, recorder: Recorder):
        from .superparams import mutation_rate, crossover_rate, extinct_rate, min_population_size, max_population_size
        self.steps += 1
        recorder.new_step(self.steps)
        self.speciate()
        self.set_fitness()

        for specie in self.species:
            recorder.record_specie(specie)
            specie.set_adjusted_fitness()

        self.species.sort(key=lambda x: x.best.adjusted_fitness, reverse=True)
        # eliminate the worst specie
        if random.random() < extinct_rate and len(self.species) > 1:
            self.species.pop(-1)

        # calculation produce rate
        population_size = len(self.organisms)
        scale = (population_size - min_population_size) / (max_population_size - min_population_size)

        if population_size <= min_population_size:
            produce_rate = 1 + random.random() * 0.5
        elif population_size >= max_population_size:
            produce_rate = random.random() * 0.5
        else:
            produce_rate = 1 - scale * random.random()

        produce_rate *= 0.5

        # generate the next generation
        new_organisms = []
        for specie in self.species:
            # elitism
            best = deepcopy(specie.best)
            new_organisms.append(best)
            best.genome_id_ = (self.steps, len(new_organisms))
            if recorder:
                recorder.record_innovation([specie.best], best)
            # calculate choice weight based on fitness
            choice_weights = [organism.adjusted_fitness for organism in specie.organisms]
            total_fitness = sum(choice_weights)
            choice_weights = [w / total_fitness for w in choice_weights]
            # crossover
            for _ in range(int(population_size * produce_rate)):
                if random.random() < crossover_rate:
                    parent1, parent2 = random.choices(specie.organisms, weights=choice_weights, k=2)
                    try:
                        child = parent1 * parent2
                        new_organisms.append(child)
                        child.genome_id_ = (self.steps, len(new_organisms))
                        if recorder:
                            recorder.record_innovation([parent1, parent2], child, innovation_type='crossover')
                    except Exception as e:
                        print(e)
            # mutation
            for _ in range(int(population_size * produce_rate)):
                if random.random() < mutation_rate:
                    organism = random.choices(specie.organisms, weights=choice_weights, k=1)[0]
                    new_organism, innovation_types = organism.mutate()
                    new_organisms.append(new_organism)
                    new_organism.genome_id_ = (self.steps, len(new_organisms))
                    if recorder:
                        recorder.record_innovation([organism], new_organism, innovation_type='|'.join(innovation_types))

        # Control population size
        new_organisms = [organism for organism in new_organisms
                          if len(organism.input_nodes) == len(self.organisms[0].input_nodes)
                          and len(organism.output_nodes) == len(self.organisms[0].output_nodes)]
        if len(new_organisms) > self.max_size:
            new_organisms = new_organisms[:self.max_size]
        self.organisms = new_organisms

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "organisms": [organism.to_dict() for organism in self.organisms],
            "steps": self.steps,
            "innovation_number": InnovationNumber.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict, fittness_func: any):
        organisms = [Genome.from_dict(organism_dict) for organism_dict in data['organisms']]
        max_size = int(data['max_size'])
        InnovationNumber.from_dict(data['innovation_number'])
        steps = int(data['steps'])

        population = cls(organisms, max_size, fittness_func)
        population.steps = steps
        return population

    def add_checkpoint(self):
        file = os.path.join(checkpoint_path, f'checkpoint_{self.steps}.json')
        with open(file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_checkpoint(cls, step: int, fittness_func: any):
        try:
            file = os.path.join(checkpoint_path, f'checkpoint_{step}.json')
            with open(file, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data, fittness_func)
        except FileNotFoundError:
            return None

