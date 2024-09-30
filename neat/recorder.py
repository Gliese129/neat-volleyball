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
        self._create_folder(self.img_folder)
        with open(self.log_file, 'w') as f:
            f.write('')

    @staticmethod
    def _create_folder(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def record_organisms(self, organisms: list[Genome], folder: str):
        folder = os.path.join(self.folder, folder)
        self._create_folder(folder)
        for organism in organisms:
            path = os.path.join(folder, f'{organism.genome_id}.png')
            if organism.genome_id not in self.networks:
                self.network_to_img(organism, path)
                self.networks[organism.genome_id] = path


    def record_specie(self, specie: Species):
        for organism in specie.organisms:
            if organism.genome_id not in self.networks:
                file = os.path.join(self.img_folder, f'{organism.genome_id}.png')
                self.network_to_img(organism, file)
                self.networks[organism.genome_id] = f'./{self.step}/{organism.genome_id}.png'
        # add log
        with open(self.log_file, 'a') as f:
            # record species
            content = [f"   [{organism.genome_id} fittness {organism.fitness}]({self.networks[organism.genome_id]})"
                       for organism in specie.organisms]
            content = '\n'.join(content)
            f.write(f'specie:\n {content}')

    def record_innovation(self, parents: list[Genome], child: Genome):
        with open(self.log_file, 'a') as f:
            if len(parents) == 1:
                content = parents[0].genome_id
            else:
                content = ' X '.join([parent.genome_id for parent in parents])
            f.write(f'innovation:\n {content} => {child.genome_id}\n')

    @staticmethod
    def network_to_img(network: Genome, file: str):
        graph = pydot.Dot(graph_type='digraph')
        node_map = {}
        for node in network.nodes.values():
            graph_node = pydot.Node(node.node_id, label=node.activation.value)
            node_map[node.node_id] = graph_node
            graph.add_node(graph_node)

        for edge in network.edges.values():
            if edge.weight != 0:
                from_ = node_map[edge.from_node]
                to_ = node_map[edge.to_node]
                graph_edge = pydot.Edge(from_, to_)
                graph.add_edge(graph_edge)

        graph.write_png(file)
