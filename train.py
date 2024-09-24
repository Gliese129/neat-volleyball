import os
import random

from fitness import get_score
from neat.activation import sigmoid
from neat.gene import Gene
from neat.genome import Genome
from neat.node import Node
from neat.population import Population
from neat.recorder import Recorder, delete_all_files_in_folder
from neat.superparams import disable_weight_rate, checkpoint_path

input_node_num = 12
output_node_num = 3
init_population_size = 80
steps = 30
random_pick_size = 5

log_path = './log'



def gen_init_genomes():
    input_nodes = [Node() for _ in range(input_node_num)]
    output_nodes = [Node() for _ in range(output_node_num)]
    init_nodes = input_nodes + output_nodes
    init_nodes[-1].activation = sigmoid
    init_genomes = []
    for _ in range(init_population_size):
        edges = []
        for input_node in input_nodes:
            for output_node in output_nodes:
                weight = random.random()
                if random.random() < disable_weight_rate:
                    weight = 0
                edges.append(Gene(input_node.node_id, output_node.node_id, weight))
        init_genomes.append(Genome(init_nodes, edges))
    return init_genomes


def fittest_func(this: Genome, all_genomes: list[Genome], this_specie: list[Genome]):
    tot_score = 0
    # all
    env_score = 0
    others = random.choices(all_genomes, k = random_pick_size)
    for rival in others:
        left_score, _ = get_score(this, rival)
        env_score += left_score
    env_score /= len(others)
    # only same specie
    specie_score = 0
    others = random.choices(this_specie, k = min(random_pick_size, len(this_specie)))
    for rival in others:
        left_score, _ = get_score(this, rival)
        specie_score += left_score
    specie_score /= len(others)
    # use baseline
    base_score, _ = get_score(this)
    return env_score + specie_score + base_score


population = Population(gen_init_genomes(), 100, fittest_func)
recorder = Recorder(log_path)

# remove all files

if __name__ == '__main__':
    if os.path.exists(log_path):
        delete_all_files_in_folder(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Start training')
    for i in range(steps):
        recorder.new_step(i)
        population.step(recorder=recorder)
        print(f'step {i}, best fitness: {population.best.fitness} size: {len(population.organisms)}')
        print(population.best)
    population.best.save('./output/best.pickle')
