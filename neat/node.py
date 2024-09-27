from .global_state import InnovationNumber
from .activation import activation_function_dict
nodes = []

class Node:
    """
    A node in a neural network. Each node has a unique id, and an activation function.
    """

    node_id: int
    activation: any
    activation_name: str

    x: float

    def __init__(self, node_id: int = None, activation_name: str = None):

        if node_id is None:
            node_id = InnovationNumber.new_node_innovation_number()
        if activation_name is None:
            activation_name = 'none'

        self.activation_name = activation_name
        self.node_id = node_id
        self.activation = activation_function_dict[activation_name]

    def __str__(self):
        return f'Node {self.node_id}'

    def to_dict(self) -> dict:
        """
        Convert the Node object to a dictionary that can be serialized to JSON.
        """
        return {
            "node_id": self.node_id,
            "activation": self.activation_name
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        """
        Create a Node object from a dictionary.
        :param data: The dictionary containing node data.
        :return: Node object
        """
        node_id = data["node_id"]
        activation_name = data["activation"]
        return cls(node_id=node_id, activation_name=activation_name)
