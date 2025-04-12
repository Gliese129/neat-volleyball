import time

import jax.numpy as jnp
from jax import random
import jax
from .p import HyperParams, activation_functions


class Individual:
    """
    Attributes:
        nodes: node genes (jnp array)
        genes: edge genes (jnp array)
        fitness: fitness value(float)
        rank: rank of individual (int)
        generation: generation number (int)
        specie: species id (int)
    """
    fitness: float
    rank: int
    generation: int
    specie: int

    def __init__(self, nodes: jnp.ndarray, genes: jnp.ndarray):
        """ Initialize individual with given genes

        *Comparing to using a Node class, jnp has better performance when using jnp arrays*

        :param nodes: [3 * N]
                    - [0, :] == Node ID
                    - [1, :] == Node Type (1=input, 2=output, 3=hidden, 4=bias)
                    - [2, :] == Activation Function ID
        :param genes: [5 * N]
                    - [0, :] == Innovation Number
                    - [1, :] == Source Node ID
                    - [2, :] == Destination Node ID
                    - [3, :] == Weight
                    - [4, :] == Enabled  #

        """
        self.nodes = nodes
        self.genes = genes

    @property
    def conn_cnt(self) -> int:
        """
        :return: Number of active connections
        """
        return int(jnp.sum(self.genes[4, :]))

    @property
    def adj_matrix(self) -> jnp.ndarray:
        """
        :return: adjacency matrix[N * N]
        """
        node_cnt = jnp.max(self.nodes[0, :]) + 1
        result = jnp.zeros((node_cnt, node_cnt), dtype=jnp.float32)

        sources = self.genes[1, self.genes[4, :] == 1].astype(jnp.int32)
        destinations = self.genes[2, self.genes[4, :] == 1].astype(jnp.int32)
        weights = self.genes[3, self.genes[4, :] == 1]

        result = result.at[sources, destinations].set(weights)
        return result

    def get_node_levels(self) -> jnp.ndarray:
        """
        :return: node levels[N]
        """
        adj_matrix = self.adj_matrix
        node_cnt = adj_matrix.shape[0]
        max_iter = node_cnt

        types = jnp.zeros(node_cnt, dtype=jnp.int32)
        types = types.at[self.nodes[0, :].astype(jnp.int32)].set(self.nodes[1, :].astype(jnp.int32))

        init_levels = jnp.where((types == 1) | (types == 4), 0.0, -1e9)

        def body_fn(levels, _):
            mask = adj_matrix != 0
            candidate_matrix = jnp.where(mask, levels[:, None] + 1.0, -1e9)
            new_levels = jnp.max(candidate_matrix, axis=0)
            new_levels = jnp.where((types == 1) | (types == 4), 0.0, new_levels)
            levels = jnp.maximum(levels, new_levels)
            return levels, None

        levels, _ = jax.lax.scan(body_fn, init_levels, None, length=max_iter)
        return levels


    def create_child(self, p: HyperParams, innovation_record: jnp.array, generation: int = 0, another: 'Individual' = None) -> ('Individual', jnp.ndarray):
        """
        Create child individual from parent individual
        :param p: HyperParams
        :param innovation_record: innovation record
        :param generation: generation number
        :param another: another individual for crossover
        :return: child
        """
        assert innovation_record is not None, "innovation_record is None"
        if another is None:
            child = Individual(self.nodes.copy(), self.genes.copy())
        else:
            child = self._crossover(another)
        innovation_record_ =child._mutate(p, innovation_record)
        child.generation = generation

        return child, innovation_record_

    def _crossover(self, another: 'Individual') -> 'Individual':
        """
        Crossover between two individuals
        Structure is based on self, and genes are randomly selected from another(only common genes)
        :param another: another individual
        :return: child
        """
        key = random.key(int(time.time()))

        a_genes = self.genes[0, :]
        b_genes = another.genes[0, :]
        _, a_common_idx, b_common_idx = jnp.intersect1d(a_genes, b_genes, return_indices=True)

        b_select_prob = 0.5
        select_mask = random.uniform(key, shape=(len(b_common_idx),)) < b_select_prob

        new_genes = jnp.copy(self.genes)
        new_genes[3, a_common_idx[select_mask]] = another.genes[3, b_common_idx[select_mask]]
        new_nodes = jnp.copy(self.nodes)

        child = Individual(new_nodes, new_genes)

        return child

    def _mutate(self, p: HyperParams, innovation_record: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """
        Mutate individual
        :param p: HyperParams
        :param innovation_record: innovation record
        :return: updated innovation record
        """
        # re-enable disabled genes
        disabled_genes = (jnp.where(self.genes[4, :] == 0))[0]
        key, subkey = random.split(key)
        enable_mask = random.uniform(subkey, shape=(len(disabled_genes),)) < p.enable_rate
        self.genes[4, disabled_genes[enable_mask]] = 1

        # mutate weight
        key, subkey = random.split(key)
        weight_mask = random.uniform(subkey, shape=(self.conn_cnt,)) < p.mutate_weight_rate
        key, subkey = random.split(key)
        self.genes[3, weight_mask] = random.uniform(subkey, shape=(self.conn_cnt,)) * 2 - 1 # allow negative weight

        # mutate activation function ? maybe not needed
        key, subkey = random.split(key)
        activation_mask = random.uniform(subkey, shape=(self.conn_cnt,)) < p.mutate_activation_rate
        key, subkey = random.split(key)
        self.nodes[2, activation_mask] = random.randint(subkey, shape=(self.conn_cnt,), minval=0, maxval=len(activation_functions))

        # mutate add node
        key, subkey = random.split(key)
        if random.uniform(subkey) < p.mutation_rate:
            innovation_record = self._mutate_add_node(p, innovation_record, key)
        # mutate add connection
        key, subkey = random.split(key)


        return innovation_record

    def _mutate_add_connection(self, p: HyperParams, innovation_record: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """
        Add connection to individual
        :param p: HyperParams
        :param innovation_record: innovation record
        :param key: random key
        :return: updated innovation record
        """
        node_levels = self.get_node_levels()
        # generate all possible connections
        exist_nodes = self.nodes[0, :].astype(jnp.int32).tolist()
        exist_nodes = jnp.ndarray(set(exist_nodes))
        levels = node_levels[exist_nodes]
        cmp_mask = jnp.expand_dims(levels, axis=1) <= jnp.expand_dims(levels, axis=0)
        idx_mask = jnp.expand_dims(exist_nodes, axis=1) != jnp.expand_dims(exist_nodes, axis=0)
        cmp_mask = jnp.logical_and(cmp_mask, idx_mask)
        valid_idx = jnp.where(cmp_mask)
        valid_connections = jnp.stack([exist_nodes[valid_idx[0]], exist_nodes[valid_idx[1]]], axis=1)
        # get all connections
        all_connections = self.genes[1:3, :].T
        # remove existing connections
        valid_connections = jnp.setdiff1d(valid_connections, all_connections, assume_unique=True)
        # select random connection
        key, subkey = random.split(key)
        conn_idx = random.choice(subkey, valid_connections.shape[0])
        selected_conn = valid_connections[conn_idx]





    def _mutate_add_node(self, p: HyperParams, innovation_record: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """
        Add node to individual
        :param p: HyperParams
        :param innovation_record: innovation record
        :return: updated innovation record
        """
        tot_node_cnt = int(jnp.max(innovation_record[2, :])) + 1
        tot_conn_cnt = innovation_record[0, -1] + 1
        # select random connection
        key, subkey = random.split(key)
        active_conn_idx = jnp.where(self.genes[4, :] == 1)[0]
        conn_idx = random.choice(subkey, active_conn_idx)
        selected_conn = self.genes[:, conn_idx]
        # create new node
        key, subkey = random.split(key)
        act_fun = random.randint(subkey, shape=(1,), minval=0, maxval=len(activation_functions))
        new_node = jnp.array([tot_node_cnt, 3, act_fun])[:, None]
        # create new connection
        new_conn_s = selected_conn.copy().at[2].set(tot_node_cnt).at[0].set(tot_conn_cnt)  # source to new node(remain weight)
        new_conn_d = selected_conn.copy().at[1].set(tot_node_cnt).at[0].set(tot_conn_cnt + 1).at[3].set(1)  # new node to destination
        # apply mutation
        self.nodes = jnp.concatenate((self.nodes, new_node[:, None]), axis=1)
        self.genes = jnp.concatenate((self.genes, new_conn_s[:, None], new_conn_d[:, None]), axis=1)
        # update innovation record
        innovation_record = jnp.concatenate((innovation_record, new_conn_s[:, None], new_conn_d[:, None]), axis=1)
        return innovation_record








