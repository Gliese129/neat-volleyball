class InnovationNumber:
    gene_innovation_number = 0
    node_innovation_number = 0

    all_genes = []

    @classmethod
    def get_gene_innovation_number(cls, from_: int, to_: int):
        gene = (from_, to_)
        if gene in cls.all_genes:
            return cls.all_genes.index(gene) + 1

        cls.gene_innovation_number += 1
        cls.all_genes.append(gene)
        return cls.gene_innovation_number

    @classmethod
    def new_node_innovation_number(cls):
        cls.node_innovation_number += 1
        return cls.node_innovation_number


    @classmethod
    def to_dict(cls) -> dict:
        """
        Convert the InnovationNumber state to a dictionary.
        """
        return {
            "gene_innovation_number": cls.gene_innovation_number,
            "node_innovation_number": cls.node_innovation_number,
            "all_genes": cls.all_genes
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Restore the InnovationNumber state from a dictionary.
        """
        cls.gene_innovation_number = data["gene_innovation_number"]
        cls.node_innovation_number = data["node_innovation_number"]
        cls.all_genes = data["all_genes"]


