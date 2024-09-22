from typing import Optional

from neat.genome import Genome
from .superparams import distance_threshold, species_best_size


class Species:
    genomes: list[Genome]
    representative: Genome
    best: Optional[Genome]

    def __init__(self, representative: Genome):
        self.representative = representative
        self.genomes = []
        self.best = None

    def distance(self, genome: Genome) -> float:
        return self.representative - genome

    def can_have(self, genome: Genome) -> bool:
        return self.distance(genome) < distance_threshold

    def set_adjusted_fitness(self):
        n = 1 if len(self.genomes) < species_best_size else len(self.genomes)
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness / n
            if self.best is None or genome.fitness > self.best.fitness:
                self.best = genome

