import random

from neat.activation import sigmoid
from neat.gene import Gene
from neat.node import Node
from neat.population import Population
from neat.genome import Genome
from neat.superparams import disable_weight_rate
from fitness import get_score

input_node_num = 12
output_node_num = 3
init_population_size = 50
steps = 10
random_pick_size = 5

def gen_init_genomes():
    init_nodes = [Node(i) for i in range(input_node_num + output_node_num)]
    init_nodes[-1].activation = sigmoid
    init_genomes = []
    for i in range(init_population_size):
        edges = []
        for j in range(input_node_num):
            for k in range(input_node_num, input_node_num + output_node_num):
                weight = random.random()
                if random.random() < disable_weight_rate:
                    weight = 0
                edges.append(Gene(j, k, weight))
        init_genomes.append(Genome(init_nodes, edges))
    return init_genomes


def fittest_func(this: Genome, all_genomes: Population):
    others = all_genomes.organisms.copy()
    others.remove(this)
    others = random.choices(others, k = random_pick_size)

    tot_score = 0
    for rival in others:
        left_score, right_score = get_score(this, rival)
        tot_score += left_score
    return tot_score / random_pick_size


population = Population(gen_init_genomes(), 100, fittest_func)

if __name__ == '__main__':
    print('Start training')
    for i in range(steps):
        population.step()
        print(f'step {i}, best fitness: {population.best.fitness} size: {len(population.organisms)}')
        print(population.best)
    population.best.save('./output/best.pickle')
