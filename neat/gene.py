from .global_state import InnovationNumber


class Gene:
    from_node: int
    to_node: int
    id: int
    weight: float

    def __init__(self, from_node: int, to_node: int, weight: float):
        assert from_node != to_node, 'The from node and the to node should not be the same'
        assert from_node >= 0 and to_node >= 0, 'The node id should be non-negative'
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

        self.id = InnovationNumber.get_gene_innovation_number(from_node, to_node)


    def __str__(self):
        return f'Edge {self.id}:  {self.from_node} -> {self.to_node} [weight: {self.weight}]'

    def to_dict(self) -> dict:
        """
        Convert the Gene object to a dictionary that can be serialized to JSON.
        """
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "id": self.id,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Gene':
        """
        Create a Gene object from a dictionary.
        """
        # Assuming the ID is provided in the dictionary, which is generated during the gene's creation.
        gene = cls(int(data["from_node"]), int(data["to_node"]), float(data["weight"]))
        gene.id = int(data["id"])
        return gene