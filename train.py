import multiprocessing as mp

mp.set_start_method("spawn", force=True)

import json
import os
from neat import Neat, HyperParams
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from score import score

p = HyperParams(
    population_size=50,
    max_generations=10, # for testing
)


neat: Neat = None
output_folder = "output"
log_times = 1
sample_num = 5
fame_num = 5
fame_agents = []

def setup():
    global neat, output_folder, log_times
    neat = Neat(p)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def train():
    key = random.PRNGKey(0)
    best_individual = None
    for generation in tqdm(range(p.max_generations)):
        print(f"Running generation: {generation}")
        key, subkey = random.split(key)
        neat.ask()
        scores = score(
            agents=neat.population,
            key=subkey,
            sample_num=sample_num,
            score_method="reward",
            extra_agents=fame_agents
        )
        # scores = jnp.ones(len(neat.population))
        neat.tell(scores)
        # Save the best individual
        if best_individual is None or neat.population[0].fitness > best_individual.fitness:
            best_individual = neat.population[0]
        if len(fame_agents) < fame_num:
            fame_agents.append(best_individual)
        else:
            # Replace the worst agent in fame_agents with the best one
            if best_individual.fitness > fame_agents[-1].fitness:
                fame_agents[-1] = best_individual
                fame_agents.sort(key=lambda x: x.fitness, reverse=True)
        # record
        print(f"Best fitness: {best_individual.fitness}")
        if generation % log_times == 0:
            # Save the population
            # Save the population to a file
            with open(f"{output_folder}/population_{generation}.json", "w") as f:
                json.dump(neat.to_json(), f, indent=4)
            with open(f"{output_folder}/best_{generation}.json", "w") as f:
                json.dump(neat.population[0].to_json(), f, indent=4)

    with open(f"{output_folder}/best.json", "w") as f:
        json.dump(best_individual.to_json(), f, indent=4)


if __name__ == '__main__':
    # params: -p population_size -g max_generations --output output_folder --log_times log_times
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--population_size', type=int, default=50)
    parser.add_argument('-g', '--max_generations', type=int, default=10)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--log_times', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=5)

    args = parser.parse_args()
    p.population_size = args.population_size
    p.max_generations = args.max_generations
    output_folder = args.output
    log_times = args.log_times
    sample_num = args.sample_num

    setup()
    train()
