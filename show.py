import json
import os

from matplotlib import pyplot as plt

folder_path = "output"

def show_fittness_curve(folder: str, max_generation: int):
    # Load all generations of populations and build mappings:
    fitness = []
    for gen in range(max_generation):
        path = os.path.join(folder, f"population_{gen}.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        fitness.append(data['population'][0]['fitness'])

    plt.plot(fitness)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Curve')
    plt.show()

if __name__ == '__main__':
    # Load all generations of populations and build mappings:
    max_generation = 100
    show_fittness_curve(folder_path, max_generation)
