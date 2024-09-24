import multiprocessing as mp
import os
import pickle
import random
from copy import deepcopy
from typing import Optional

from neat.genome import Genome
from neat.recorder import Recorder
from neat.species import Species
from neat.superparams import checkpoint_path, checkpoint_rate


def compute_fitness(args):
    organism, all_organisms, specie_organisms, fitness_func = args
    fitness = fitness_func(organism, all_organisms, specie_organisms)
    return organism, fitness


class Population:
    max_size: int
    species: list[Species]
    organisms: list[Genome]
    best: Optional[Genome]
    steps: int = 0
    fitness_func: any

    def __init__(self, init_organisms: list[Genome], max_size: int, fittness_func):
        self.max_size = max_size
        self.organisms = init_organisms
        for idx, organism in enumerate(self.organisms):
            organism.generation = 0
            organism.idx = idx

        self.species = []
        self.fitness_func = fittness_func


    def speciate(self):
        for organism in self.organisms:
            for specie in self.species:
                if specie.can_have(organism):
                    specie.genomes.append(organism)
                    break
            else:
                self.species.append(Species(organism))

    def set_fitness(self):
        train_set = []
        idx_map = {}
        for specie in self.species:
            for organism in specie.genomes:
                train_set.append((organism, self.organisms, specie.genomes, self.fitness_func))
                idx_map[organism.idx] = organism

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(compute_fitness, train_set)

        for organism, fitness in results:
            idx_map[organism.idx].fitness = fitness

        self.organisms.sort(key=lambda x: x.fitness, reverse=True)
        self.best = self.organisms[0]

    def step(self, recorder: Recorder):
        from .superparams import mutation_rate, crossover_rate, species_best_size
        self.steps += 1
        recorder.new_step(self.steps)
        self.speciate()
        self.set_fitness()

        for specie in self.species:
            recorder.record_specie(specie)
        # generate the next generation
        new_organisms = []
        for specie in self.species:
            specie.set_adjusted_fitness()
            # eliminate the worst organisms
            specie.genomes.sort(key=lambda x: x.fitness, reverse=True)
            specie.genomes = specie.genomes[:species_best_size]
            # elitism
            best = deepcopy(specie.best)
            best.generation = self.steps
            best.idx = len(new_organisms)
            new_organisms.append(best)
            # crossover
            for _ in range(species_best_size):
                choice_weights = [genome.adjusted_fitness for genome in specie.genomes]
                sum_ = sum(choice_weights)
                choice_weights = [w / sum_ for w in choice_weights]
                if random.random() < crossover_rate:
                    parent1, parent2 = random.choices(specie.genomes, weights=choice_weights, k=2)
                    try:
                        child = parent1 * parent2
                        child.generation = self.steps
                        child.idx = len(new_organisms)
                        if recorder:
                            recorder.record_crossover(parent1, parent2, child)
                        new_organisms.append(child)
                    except Exception as e:
                        print(e)
                if random.random() < mutation_rate:
                    organism = random.choices(specie.genomes, weights=choice_weights, k=1)[0]
                    new_organism = organism.mutate()
                    new_organism.generation = self.steps
                    new_organism.idx = len(new_organisms)
                    new_organisms.append(new_organism)
        self.organisms = [organism for organism in new_organisms
                          if len(organism.input_nodes) == len(self.organisms[0].input_nodes)
                          and len(organism.output_nodes) == len(self.organisms[0].output_nodes)]
        recorder.record_organisms(self.organisms, 'debug') # debug
        self.species = []
        self.organisms = self.organisms[:self.max_size]

        self.add_checkpoint()

    def add_checkpoint(self):
        if self.steps % checkpoint_rate:
            return
        file = os.path.join(checkpoint_path, f'checkpoint_{self.steps}.pickle')
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, step: int):
        try:
            file = os.path.join(checkpoint_path, f'checkpoint_{step}.pickle')
            with open(file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None




