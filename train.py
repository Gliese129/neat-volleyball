import os
import random

from score import get_score
from neat.activation import ActivationFunction
from neat.gene import Gene
from neat.genome import Genome
from neat.node import Node
from neat.population import Population
from neat.recorder import Recorder, delete_all_files_in_folder
from neat.superparams import checkpoint_path, game_step_growth_rate, base_game_step

input_node_num = 12
output_node_num = 3
init_population_size = 100
steps = 80
random_pick_size = 3
resume = 38
log_path = './log'


def gen_init_genomes():
    input_nodes = [Node(activation=ActivationFunction.NONE) for _ in range(input_node_num)]
    output_nodes = [Node() for _ in range(output_node_num)]
    init_nodes = input_nodes + output_nodes
    init_genomes = []
    for _ in range(init_population_size):
        edges = []
        for input_node in input_nodes:
            for output_node in output_nodes:
                weight = random.random()
                if random.random() < 0.5:
                    weight = 0
                edges.append(Gene(input_node.node_id, output_node.node_id, weight))
        organism = Genome(init_nodes, edges)
        init_genomes.append(organism)
        organism.genome_id_ = (0, len(init_genomes))
    return init_genomes


def fittest_func(this: Genome, all_genomes: list[Genome], this_specie: list[Genome], current_step = None):
    max_game_step = base_game_step
    if current_step is not None:
        # as time pass, game step should be longer
        max_game_step *= game_step_growth_rate ** (current_step // 5)
    # all
    env_score = 0
    others = random.sample(all_genomes, k = random_pick_size)
    for rival in others:
        left_score, _ = get_score(this, rival, max_game_step)
        env_score += left_score
    env_score /= len(others)
    # only same specie
    specie_score = 0
    others = random.sample(this_specie, k = min(random_pick_size, len(this_specie)))
    for rival in others:
        left_score, _ = get_score(this, rival, max_game_step)
        specie_score += left_score
    specie_score /= len(others)
    # use baseline
    base_score, _ = get_score(this, None, max_game_step)
    return env_score + specie_score + base_score


population: Population = None
recorder: Recorder = None
def init():
    global population, recorder
    if resume:
        population = Population.load_checkpoint(resume, fittness_func=fittest_func)
    else:
        population = Population(gen_init_genomes(), 150, fittness_func=fittest_func)

    recorder = Recorder(log_path)

# remove all files

if __name__ == '__main__':
    input('press enter to start')

    init()
    if os.path.exists(log_path) and not resume:
        delete_all_files_in_folder(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    print('Start training')
    population.steps = resume
    for i in range(resume, steps):
        recorder.new_step(i + 1)
        population.step(recorder=recorder)
        print(f'step {i + 1}, best fitness: {population.best.fitness} size: {len(population.organisms)}')
        if i % 5 == 0:
            population.add_checkpoint()
        population.best.save(f'./output/best_{i + 1}.json')

    population.close()
    population.best.save('./output/best.json')
