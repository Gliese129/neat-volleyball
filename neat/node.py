from neat import InnovationNumber
from .activation import none_
nodes = []

class Node:
    """
    A node in a neural network. Each node has a unique id, and an activation function.
    """

    node_id: int
    activation: any

    x: float

    def __init__(self, node_id: int = None, activation = None):

        if node_id is None:
            node_id = InnovationNumber.new_node_innovation_number()
        if activation is None:
            activation = none_

        self.node_id = node_id
        self.activation = activation

    def __str__(self):
        return f'Node {self.node_id}'

