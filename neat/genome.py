import random

import jax.numpy as jnp

from .gene import Gene
from .node import Node
import pickle


class Genome:
    edges: dict[int, Gene]
    nodes: dict[int, Node]
    topology: list[list[Node]]
    depth_map: dict[int, int]
    fitness: float = 0
    adjusted_fitness: float = 0

    @property
    def cell_num(self): return len(self.nodes)
    @property
    def edge_num(self): return len(self.edges)


    def __init__(self, nodes: list[Node] | dict[int, Node], edges: list[Gene] | dict[int, Gene]):
        if isinstance(nodes, list):
            nodes = {node.node_id: node for node in nodes}
        if isinstance(edges, list):
            edges = {edge.id: edge for edge in edges}

        self.nodes = nodes
        self.edges = edges
        self.calculate_node_topology()

    def calculate_node_topology(self):
        """
        Return the depth of each node in the network
        :return: A list of tuples, each tuple contains the node id and its depth, index from 0
        """
        self.topology = []
        self.depth_map = {}
        idx = 0

        nodes = [node for node in self.nodes.values()]
        excluded = {node.node_id: False for node in nodes}

        cnt = len(nodes) * 2
        while len(nodes) > 0 and cnt > 0:
            cnt -= 1
            self.topology.append([])
            in_degree = [0 for _ in range(len(self.nodes))] # calculate the in and out degree of each node
            for edge in self.edges.values():
                if excluded[edge.from_node]: continue
                in_degree[edge.to_node] += 1
                self.nodes[edge.to_node].from_edges.append(edge)

            next_nodes = []
            # find the nodes with in_degree 0
            for node in nodes:
                if excluded[node.node_id]: continue
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
        inputs = [node for node in self.topology[0]]
        assert len(inputs) == x.shape[0], 'The input size does not match the network input size'
        for i, node in enumerate(inputs):
            node.x = x[i]

        for layer in self.topology[1:]:
            for node in layer:
                node.forward(self.nodes)

        x = [node.x for node in self.topology[-1]]
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
        nodes = {**parent1.nodes, **parent2.nodes}
        #
        edges = {}
        max_edge_id = max(parent1.edges.keys() | parent2.edges.keys()) + 1
        # calculate mismatched edges
        for i in range(max_edge_id):
            edge1 = parent1.edges.get(i)
            edge2 = parent2.edges.get(i)
            if edge1 is not None and edge2 is not None:
                edges[i] = random.choice([edge1, edge2])
                continue
            if edge1 is not None:
                edges[i] = edge1
            elif edge2 is not None:
                edges[i] = edge2
        return cls(nodes, edges)

    def mutate(self) -> 'Genome':
        from .superparams import add_edge_rate, add_node_rate, change_weight_rate, disable_weight_rate
        new_genome = Genome(self.nodes.copy(), self.edges.copy())
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
            from_node = random.choice(list(self.nodes.keys()))
            to_node = random.choice(list(self.nodes.keys()))
            if from_node == to_node:
                continue
            if self.depth_map[from_node] >= self.depth_map[to_node]:
                # make sure the edge is feed forward
                continue
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
        new_node = Node(len(self.nodes))
        self.nodes[new_node.node_id] = new_node
        new_edge = Gene(from_node, new_node.node_id, random.random())
        self.edges[new_edge.id] = new_edge
        new_edge = Gene(new_node.node_id, to_node, random.random())
        self.edges[new_edge.id] = new_edge
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

    def save(self, file: str):
        with open(file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file: str) -> 'Genome':
        with open(file, "rb") as f:
            res = pickle.load(f)
        return res





