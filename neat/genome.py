import json
import random
from collections import deque
from copy import deepcopy
from typing import List, Dict, Tuple

import jax.numpy as jnp

from .gene import Gene
from .node import Node
from .global_state import InnovationNumber


class Genome:
    edges: Dict[int, Gene]
    nodes: Dict[int, Node]
    topology: List[List[Node]]
    depth_map: Dict[int, int]
    fitness: float = 0
    adjusted_fitness: float = 0
    output_nodes: List[Node]
    input_nodes: List[Node]
    genome_id_: Tuple[int, int]

    @property
    def genome_id(self):
        return f'{self.genome_id_[0]}_{self.genome_id_[1]}'


    def __init__(self, nodes: List[Node] | Dict[int, Node], edges: List[Gene] | Dict[int, Gene],
                 generation: int = None, idx: int = None):

        if isinstance(nodes, list):
            nodes = {node.node_id: node for node in nodes}
        if isinstance(edges, list):
            edges = {edge.id: edge for edge in edges}

        self.nodes = nodes
        self.edges = edges
        self.calculate_node_topology()
        self.genome_id_ = (generation, idx)

    def calculate_node_topology(self):
        """
        Return the depth of each node in the network and calculate input/output nodes.
        :return: Updates topology and depth_map
        """
        self.topology = []
        self.depth_map = {}
        self.input_nodes = []
        self.output_nodes = []

        nodes = [node for node in self.nodes.values()]
        in_degree = {node.node_id: 0 for node in nodes}
        out_degree = {node.node_id: 0 for node in nodes}

        # Calculate in-degrees and out-degrees of all nodes
        for edge in self.edges.values():
            in_degree[edge.to_node] += 1
            out_degree[edge.from_node] += 1

        # Identify input nodes (with in-degree 0) and output nodes (with out-degree 0)
        for node in nodes:
            if in_degree[node.node_id] == 0:
                self.input_nodes.append(node)
            if out_degree[node.node_id] == 0:
                self.output_nodes.append(node)

        # Initialize a queue for topological sorting
        queue = deque([node for node in nodes if in_degree[node.node_id] == 0])

        idx = 0  # Initialize depth index
        max_depth = len(nodes)  # Maximum depth of the network
        while queue and idx < max_depth:
            current_level = []  # Collect nodes at the current depth level
            for _ in range(len(queue)):
                node = queue.popleft()
                current_level.append(node)
                self.depth_map[node.node_id] = idx

                # Decrease in-degree for each neighboring node and add to queue if in-degree becomes 0
                for edge in self.edges.values():
                    if edge.from_node == node.node_id:
                        in_degree[edge.to_node] -= 1
                        if in_degree[edge.to_node] == 0:
                            queue.append(self.nodes[edge.to_node])

            if current_level:
                self.topology.append(current_level)
            idx += 1

        # Check if there are any remaining nodes with non-zero in-degrees (i.e., cycles)
        assert all(in_degree[node.node_id] == 0 for node in nodes), ValueError(f'The network contains a cycle: {self.edges.values()}')


    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward the input through the network
        :param x: The input
        :return: The output
        """
        assert len(self.input_nodes) == x.shape[0], 'The input size does not match the network input size'

        for node in self.nodes.values():
            node.x = None
            node.from_nodes = []
        for i, node in enumerate(self.input_nodes):
            node.x = x[i]

        # init
        for edge in self.edges.values():
            from_, weight_, to_ = edge.from_node, edge.weight, edge.to_node
            self.nodes[to_].from_nodes.append((self.nodes[from_], weight_))

        # forward
        for node in self.output_nodes:
            node.forward()

        x = [node.x for node in self.output_nodes]
        return jnp.array(x).flatten()

    def distance(self, another: 'Genome'):
        """
        Calculate the distance between two networks
        :param another: The other network
        :return: The distance
        """
        from .superparams import c1, c2, c3
        disjoint = 0
        excess = 0
        weight_diff = 0
        max_edge_id = max(self.edges.keys() | another.edges.keys())
        for i in range(max_edge_id):
            edge1 = self.edges.get(i)
            edge2 = another.edges.get(i)
            if edge1 is None and edge2 is None:
                continue
            if edge1 is None or edge2 is None:
                if i <= min(max(self.edges.keys()), max(another.edges.keys())):
                    disjoint += 1
                else:
                    excess += 1
                continue
            weight_diff += abs(edge1.weight - edge2.weight)
        disjoint /= max(1, len(self.edges) + len(another.edges))
        excess /= max(1, len(self.edges) + len(another.edges))
        weight_diff /= max(1, len(self.edges) + len(another.edges))
        return c1 * disjoint + c2 * excess + c3 * weight_diff

    @classmethod
    def crossover(cls, parent1: 'Genome', parent2: 'Genome') -> 'Genome':
        """
        Crossover two networks
        :param parent1: The first parent network
        :param parent2: The second parent network
        :return: A new network
        """
        # nodes should be the union of the two parents
        edges = {}
        nodes_needed = set()

        all_edge_ids = set(parent1.edges.keys()).union(parent2.edges.keys())

        # calculate mismatched edges
        for edge_id in all_edge_ids:
            edge1 = parent1.edges.get(edge_id)
            edge2 = parent2.edges.get(edge_id)

            if edge1 and edge2:
                chosen_edge = random.choice([edge1, edge2])
            elif edge1:
                chosen_edge = edge1
            elif edge2:
                chosen_edge = edge2
            else:
                continue

            new_edge = deepcopy(chosen_edge)
            edges[edge_id] = new_edge
            nodes_needed.update([new_edge.from_node, new_edge.to_node])

        nodes = {}
        for node_id in nodes_needed:
            parent_ = [parent1.nodes.get(node_id), parent2.nodes.get(node_id)]
            parent_ = [node for node in parent_ if node]
            if parent_:
                node = random.choice(parent_)
                nodes[node_id] = deepcopy(node)
            else:
                raise ValueError(f"Node {node_id} not found in parents.")

        return cls(nodes, edges)

    def mutate(self) -> 'Genome':
        from .superparams import add_edge_rate, add_node_rate, change_weight_rate, disable_weight_rate
        new_genome = Genome(deepcopy(self.nodes), deepcopy(self.edges))

        if random.random() < add_edge_rate:
            new_genome._add_edge()
        if random.random() < add_node_rate:
            new_genome._add_node()
        if random.random() < change_weight_rate:
            new_genome._change_weight()
        if random.random() < disable_weight_rate:
            new_genome._disable_weight()
        return new_genome

    def _add_edge(self):
        cnt = 0
        while cnt < 30:
            cnt += 1
            from_node, to_node = random.choices(list(self.nodes.keys()), k=2)

            # make sure the edge is feed forward
            if self.depth_map[from_node] >= self.depth_map[to_node]:
                continue
            # make sure the edge doesn't exist
            if any(edge.from_node == from_node and edge.to_node == to_node for edge in self.edges.values()):
                continue

            new_edge = Gene(from_node, to_node, random.random())
            self.edges[new_edge.id] = new_edge
            break

        self.calculate_node_topology()

    def _add_node(self):
        edge = random.choice(list(self.edges.values()))
        from_node = edge.from_node
        to_node = edge.to_node
        new_node = Node(randomize=True)

        new_edge_1 = Gene(from_node, new_node.node_id, random.random())
        new_edge_2 = Gene(new_node.node_id, to_node, random.random())

        self.nodes[new_node.node_id] = new_node
        self.edges[new_edge_1.id] = new_edge_1
        self.edges[new_edge_2.id] = new_edge_2

        self.edges[edge.id].weight = 0 # disable the old edge

        self.calculate_node_topology()

    def _change_weight(self):
        edge = random.choice(list(self.edges.values()))
        edge.weight = random.random()

    def _disable_weight(self):
        edge = random.choice(list(self.edges.values()))
        edge.weight = 0


    def __mul__(self, other):
        return Genome.crossover(self, other)

    def __sub__(self, other):
        return self.distance(other)

    def to_dict(self) -> dict:
        """
        Convert the Genome object to a dictionary that can be serialized to JSON.
        """
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "id": self.genome_id_,
            "input_nodes": [node.node_id for node in self.input_nodes],
            "output_nodes": [node.node_id for node in self.output_nodes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        """
        Create a Genome object from a dictionary.
        """
        nodes = [Node.from_dict(node_data) for node_data in data["nodes"]]
        edges = [Gene.from_dict(edge_data) for edge_data in data["edges"]]

        genome = cls(nodes, edges, generation=data["id"][0], idx=data["id"][1])
        genome.input_nodes = [genome.nodes[node_id] for node_id in data["input_nodes"]]
        genome.output_nodes = [genome.nodes[node_id] for node_id in data["output_nodes"]]
        return genome

    def save(self, file: str):
        """
        Save the Genome object to a JSON file.
        """
        with open(file, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(file: str) -> 'Genome':
        """
        Load a Genome object from a JSON file.
        """
        with open(file, "r") as f:
            data = json.load(f)
        return Genome.from_dict(data)
