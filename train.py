import json

from neat import Neat, HyperParams
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
from score import score

p = HyperParams(
    population_size=50,
    max_generations=10, # for testing
)

neat = Neat(p)
output_folder = "output"
log_times = 2


def train():
    key = random.PRNGKey(0)
    best_individual = None
    for generation in tqdm(range(p.max_generations)):
        print(f"Generation: {generation}")
        key, subkey = random.split(key)
        neat.ask()
        print(f"running scoring for generation {generation}")
        scores = score(
            agents=neat.population,
            key=subkey,
            sample_num=2,
            score_method="reward",
        )
        # scores = jnp.ones(len(neat.population))
        print(f"scoring done for generation {generation}")
        print(f"Scores: {scores}")
        neat.tell(scores)
        # Save the best individual
        if best_individual is None or neat.population[0].fitness > best_individual.fitness:
            best_individual = neat.population[0]
        # record
        if generation % log_times == 0:
            # Save the population
            print(f"Saving population for generation {generation}")
            # Save the population to a file
            with open(f"{output_folder}/population_{generation}.json", "w") as f:
                json.dump(neat.to_json(), f, indent=4)
            with open(f"{output_folder}/best_{generation}.json", "w") as f:
                json.dump(neat.population[0].to_json(), f, indent=4)

    with open(f"{output_folder}/best.json", "w") as f:
        json.dump(best_individual.to_json(), f, indent=4)


if __name__ == '__main__':
    train()
