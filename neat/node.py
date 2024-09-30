from typing import List, Tuple

from .global_state import InnovationNumber
from .activation import ActivationFunction

import jax.numpy as jnp

class Node:
    """
    A node in a neural network. Each node has a unique id, and an activation function.
    """

    node_id: int
    activation: ActivationFunction

    x: float
    from_nodes: List[Tuple['Node', float]] # (from_node, weight)

    def __init__(self, node_id: int = None, activation: ActivationFunction = None, randomize: bool = True):
        if node_id is None:
            node_id = InnovationNumber.new_node_innovation_number()
        if activation is None:
            if randomize:
                activation = ActivationFunction.random()
            else:
                activation = ActivationFunction.NONE

        self.node_id = node_id
        self.activation = activation

    def __str__(self):
        return f'Node {self.node_id}'

    def forward(self) -> jnp.array:
        if self.x is not None:
            return self.x
        x = jnp.array([from_node.forward() * weight for from_node, weight in self.from_nodes])
        self.x = self.activation(x.sum()).tolist()
        return self.x

    def to_dict(self) -> dict:
        """
        Convert the Node object to a dictionary that can be serialized to JSON.
        """
        return {
            "node_id": self.node_id,
            "activation": self.activation.value
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        """
        Create a Node object from a dictionary.
        :param data: The dictionary containing node data.
        :return: Node object
        """
        node_id = int(data["node_id"])
        activation = data["activation"]
        activation = ActivationFunction.from_str(activation)
        return cls(node_id=node_id, activation=activation)
