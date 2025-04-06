import jax.numpy as jnp

class Individual:
    """
    Attributes:
        nodes: node genes (jnp array)
        edges: edge genes (jnp array)
        fitness: fitness value(float)
        rank: rank of individual (int)
        generation: generation number (int)
        specie: species id (int)
    """
    fitness: float
    rank: int
    generation: int
    specie: int

    def __init__(self, nodes: jnp.ndarray, edges: jnp.ndarray):
        """ Initialize individual with given genes

        *Comparing to using a Node class, jnp has better performance when using jnp arrays*

        :param nodes: [3 * N]
                    - [0, :] == Node ID
                    - [1, :] == Node Type (1=input, 2=output, 3=hidden, 4=bias)
                    - [2, :] == Activation Function ID
        :param edges: [5 * N]
                    - [0, :] == Innovation Number
                    - [1, :] == Source Node ID
                    - [2, :] == Destination Node ID
                    - [3, :] == Weight
                    - [4, :] == Enabled  #

        """
        self.nodes = nodes
        self.edges = edges

    @property
    def conn_cnt(self) -> int:
        """
        :return: Number of active connections
        """
        return int(jnp.sum(self.edges[4, :]))