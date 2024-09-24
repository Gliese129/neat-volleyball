import os
import shutil

import pydot

from neat.genome import Genome
from neat.species import Species


def delete_all_files_in_folder(folder_path):
    shutil.rmtree(folder_path)


class Recorder:
    folder: str

    # change by step
    log_file: str
    networks: dict
    img_folder: str
    step: int


    def __init__(self, folder):
        self.folder = folder
        if not os.path.exists(folder):
            os.makedirs(folder)

    def new_step(self, step: int):
        # generate img files
        self.log_file = os.path.join(self.folder, f'recorder_{step}.md')
        self.img_folder = os.path.join(self.folder, str(step))
        self.networks = dict() # organism -> img
        self.step = step
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)
        with open(self.log_file, 'w') as f:
            f.write('')

    def record_organisms(self, organisms: list[Genome], folder: str):
        folder = os.path.join(self.folder, folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
        for organism in organisms:
            path = os.path.join(folder, f'{organism.genome_id}.png')
            self.network_to_img(organism, path)


    def record_specie(self, specie: Species):
        for organism in specie.genomes:
            file = os.path.join(self.img_folder, f'{organism.genome_id}.png')
            self.networks[organism.genome_id] = f'./{self.step}/{organism.genome_id}.png'
            self.network_to_img(organism, file)
        # add log
        with open(self.log_file, 'a') as f:
            # record species
            content = []
            for genome in specie.genomes:
                content.append(f"   [{genome.genome_id} fittness {genome.fitness}]({self.networks[genome.genome_id]})")
            content = '\n'.join(content)
            content = f'specie:\n {content}\n'
            f.write(content)

    def record_crossover(self, parent1: Genome, parent2: Genome, child: Genome):
        child_img = os.path.join(self.img_folder, 'child_' + child.genome_id + '.png')
        self.network_to_img(child, child_img)
        with open(self.log_file, 'a') as f:
            content = (f'crossover:\n '
                       f'![parent1]({self.networks[parent1.genome_id]}) X '
                       f'![parent2]({self.networks[parent2.genome_id]}) =>\n '
                       f'![child](./{self.step}/child_{child.genome_id}.png) \n'
                       f'-------------------\n')
            f.write(content)

    @staticmethod
    def network_to_img(network: Genome, file: str):
        graph = pydot.Dot(graph_type='digraph')
        node_map = {}
        for node in network.nodes.values():
            graph_node = pydot.Node(node.node_id)
            node_map[node.node_id] = graph_node
            graph.add_node(graph_node)

        for edge in network.edges.values():
            if edge.weight != 0:
                from_ = node_map[edge.from_node]
                to_ = node_map[edge.to_node]
                graph_edge = pydot.Edge(from_, to_)
                graph.add_edge(graph_edge)

        graph.write_png(file)