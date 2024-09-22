from neat.gene import Gene
import jax.numpy as jnp


class Node:
    """
    A node in a neural network. Each node has a unique id, and an activation function.
    """

    node_id: int
    activation: any
    from_edges: list[Gene]

    x: float

    def __init__(self, node_id: int, activation = None):
        self.node_id = node_id
        if activation is None:
            self.activation = lambda x: x
        else:
            self.activation = activation
        self.from_edges = []
        self.x = 0.0

    def forward(self, nodes: dict[int, 'Node']) -> float:
        for edge in self.from_edges:
            self.x += edge.weight * nodes[edge.from_node].forward(nodes)
        self.x = self.activation(self.x)
        return self.x

    def __str__(self):
        return f'Node {self.node_id}'

