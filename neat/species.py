import math
from typing import Optional, List

from .genome import Genome
from .superparams import distance_threshold, specie_best_size


class Species:
    organisms: List[Genome]
    representative: Genome
    best: Optional[Genome]

    def __init__(self, representative: Genome):
        self.representative = representative
        self.organisms = [representative]
        self.best = None

    def distance(self, genome: Genome) -> float:
        return self.representative - genome

    def can_have(self, genome: Genome) -> bool:
        return self.distance(genome) < distance_threshold

    def set_adjusted_fitness(self):
        if len(self.organisms) <= specie_best_size:
            n = 1
        else:
            # When size exceeds specie_best_size, n grows exponentially
            excess = len(self.organisms) - specie_best_size
            n = math.exp(excess ** 0.01)
        for genome in self.organisms:
            genome.adjusted_fitness = genome.fitness / n
            if self.best is None or genome.fitness > self.best.fitness:
                self.best = genome

