import json
import pickle
import random
from copy import deepcopy

import jax.numpy as jnp

from .gene import Gene
from .node import Node


class Genome:
    edges: dict[int, Gene]
    nodes: dict[int, Node]
    topology: list[list[Node]]
    depth_map: dict[int, int]
    fitness: float = 0
    adjusted_fitness: float = 0
    output_nodes: list[Node]
    input_nodes: list[Node]
    generation: int
    idx: int
    activation_name: str

    @property
    def cell_num(self): return len(self.nodes)
    @property
    def edge_num(self): return len(self.edges)
    @property
    def genome_id(self): return f'{self.generation}_{self.idx}'


    def __init__(self, nodes: list[Node] | dict[int, Node], edges: list[Gene] | dict[int, Gene], generation: int = None, idx: int = None, activation_name: str = None):
        if isinstance(nodes, list):
            nodes = {node.node_id: node for node in nodes}
        if isinstance(edges, list):
            edges = {edge.id: edge for edge in edges}

        self.nodes = nodes
        self.edges = edges
        self.calculate_node_topology()
        self.generation = generation
        self.idx = idx
        self.activation_name = activation_name

    def calculate_node_topology(self):
        """
        Return the depth of each node in the network
        :return: A list of tuples, each tuple contains the node id and its depth, index from 0
        """
        self.topology = []
        self.depth_map = {}
        self.input_nodes = []
        self.output_nodes = []
        idx = 0

        nodes = [node for node in self.nodes.values()]
        excluded = {node.node_id: False for node in nodes}
        cnt = len(nodes) * 2

        in_degree = {node.node_id: 0 for node in nodes}  # calculate the in and out degree of each node
        out_degree = {node.node_id: 0 for node in nodes}
        for edge in self.edges.values():
            in_degree[edge.to_node] += 1
            out_degree[edge.from_node] += 1
        for node in nodes:
            if in_degree[node.node_id] == 0:
                self.input_nodes.append(node)
            if out_degree[node.node_id] == 0:
                self.output_nodes.append(node)


        while nodes and cnt > 0:
            cnt -= 1
            self.topology.append([])
            in_degree = {node.node_id: 0 for node in nodes} # calculate the in and out degree of each node

            # init from_edges
            for node in nodes:
                node.from_edges = []

            for edge in self.edges.values():
                if excluded.get(edge.from_node, False):
                    continue
                in_degree[edge.to_node] += 1


            next_nodes = []

            # find the nodes with in_degree 0
            for node in nodes:
                if excluded[node.node_id]:
                    continue
                if in_degree[node.node_id] == 0:
                    self.topology[-1].append(node)
                    self.depth_map[node.node_id] = idx
                    excluded[node.node_id] = True
                else:
                    next_nodes.append(node)
            # remove the nodes with in_degree 0
            nodes = next_nodes
            idx += 1

        assert cnt > 0, f'The network contains a cycle {self.edges.values()}'


    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward the input through the network
        :param x: The input
        :return: The output
        """
        assert len(self.input_nodes) == x.shape[0], 'The input size does not match the network input size'
        for i, node in enumerate(self.input_nodes):
            node.x = x[i]

        # init
        for layer in self.topology[1:]:
            for node in layer:
                node.x = 0

        net_map = {}
        for edge in self.edges.values():
            if not net_map.get(edge.from_node, False):
                net_map[edge.from_node] = list()
            net_map[edge.from_node].append(edge)

        # forward
        for layer in self.topology:
            for node in layer:
                for edge in net_map.get(node.node_id, []):
                    to_node = self.nodes[edge.to_node]
                    to_node.x += node.x * edge.weight


        x = [node.x for node in self.output_nodes]
        return jnp.array(x)

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
            node = parent1.nodes.get(node_id) or parent2.nodes.get(node_id)
            if node:         
                nodes[node_id] = deepcopy(node)
            else:
                raise ValueError(f"Node {node_id} not found in parents.")

        return cls(nodes, edges, activation_name=parent1.activation_name)

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

            print(f'{from_node} -> {to_node}')

            new_edge = Gene(from_node, to_node, random.random())
            self.edges[new_edge.id] = new_edge
            break

        self.calculate_node_topology()

    def _add_node(self):
        edge = random.choice(list(self.edges.values()))
        from_node = edge.from_node
        to_node = edge.to_node
        new_node = Node(activation_name=self.activation_name)

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
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            "generation": self.generation,
            "idx": self.idx,
            "input_nodes": [node.node_id for node in self.input_nodes],
            "output_nodes": [node.node_id for node in self.output_nodes]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Genome':
        """
        Create a Genome object from a dictionary.
        """
        nodes = {node_id: Node.from_dict(node_data) for node_id, node_data in data["nodes"].items()}
        edges = {edge_id: Gene.from_dict(edge_data) for edge_id, edge_data in data["edges"].items()}

        genome = cls(nodes, edges, generation=data["generation"], idx=data["idx"])
        genome.input_nodes = [nodes[node_id] for node_id in data["input_nodes"]]
        genome.output_nodes = [nodes[node_id] for node_id in data["output_nodes"]]
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









