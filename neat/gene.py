from neat import InnovationNumber


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