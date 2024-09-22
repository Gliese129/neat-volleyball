genes = []

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
        if not hash(self) in genes:
            genes.append(hash(self))
            self.id = len(genes) - 1
        else:
            self.id = genes.index(hash(self))

    def __hash__(self):
        return self.from_node * 100000 + self.to_node

    def __str__(self):
        return f'Edge {self.from_node} -> {self.to_node} [weight: {self.weight}]'