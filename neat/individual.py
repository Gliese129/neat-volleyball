import json

import jax.numpy as jnp
from jax import random
import jax
from .p import HyperParams, activation_functions


class Individual:
    """
    Definitions:
        N: number of nodes
        M: number of connections
        N_all: number of all nodes
        E: number of excess genes
        D: number of disjoint genes
        W: weight genes
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
        :param genes: [5 * M]
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
        :return: adjacency matrix[N_all * N_all]
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
        :return: node levels[N_all]
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


    def create_child(self, p: HyperParams, innovation_record: jnp.array, generation: int = 0, other: 'Individual' = None) -> ('Individual', jnp.ndarray):
        """
        Create child individual from parent individual
        :param p: HyperParams
        :param innovation_record: innovation record
        :param generation: generation number
        :param other: the other individual for crossover
        :return: child
        """
        key = random.PRNGKey(0)
        assert innovation_record is not None, "innovation_record is None"
        if other is None:
            child = Individual(self.nodes.copy(), self.genes.copy())
        else:
            child = self._crossover(other, key)

        innovation_record_ =child._mutate(p, innovation_record, key)
        child.generation = generation

        return child, innovation_record_

    def _crossover(self, other: 'Individual', key: jnp.ndarray) -> 'Individual':
        """
        Crossover between two individuals
        Structure is based on self, and genes are randomly selected from other(only common genes)
        Note: different from the original essay, but it is easier to write ? maybe change back later
        :param other: the other individual
        :return: child
        """
        a_genes = self.genes[0, :]
        b_genes = other.genes[0, :]
        _, a_common_idx, b_common_idx = jnp.intersect1d(a_genes, b_genes, return_indices=True)

        b_select_prob = 0.5
        select_mask = random.uniform(key, shape=(len(b_common_idx),)) < b_select_prob

        new_nodes = jnp.copy(self.nodes)
        new_genes = jnp.copy(self.genes) \
                    .at[3, a_common_idx[select_mask]].set(other.genes[3, b_common_idx[select_mask]])

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
        self.genes = self.genes.at[4, disabled_genes[enable_mask]].set(1)

        # mutate weight
        n_connections = self.genes.shape[1]
        key, subkey = random.split(key)
        weight_mask = random.uniform(subkey, shape=(n_connections,)) < p.mutate_weight_rate
        key, subkey = random.split(key)
        self.genes = self.genes.at[3, weight_mask].set(random.uniform(subkey, shape=(weight_mask.sum().item(),)) * 2 - 1) # allow negative weight

        # mutate activation function ? maybe not needed
        n_nodes = self.nodes.shape[1]
        key, subkey = random.split(key)
        activation_mask = random.uniform(subkey, shape=(n_nodes,)) < p.mutate_activation_rate
        key, subkey = random.split(key)
        self.nodes = self.nodes.at[2, activation_mask].set(
            random.randint(subkey, shape=(activation_mask.sum().item(),), minval=0, maxval=len(activation_functions) - 1) # allow negative weight
        )

        # mutate add node
        key, subkey = random.split(key)
        if random.uniform(subkey) < p.mutate_add_node_rate:
            innovation_record = self._mutate_add_node(p, innovation_record, key)
        # mutate add connection
        key, subkey = random.split(key)
        if random.uniform(subkey) < p.mutate_add_edge_rate:
            innovation_record = self._mutate_add_connection(p, innovation_record, key)
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
        exist_nodes = list(set(exist_nodes))
        exist_nodes = jnp.array(exist_nodes)
        levels = node_levels[exist_nodes]
        cmp_mask = jnp.expand_dims(levels, axis=1) <= jnp.expand_dims(levels, axis=0)
        idx_mask = jnp.expand_dims(exist_nodes, axis=1) != jnp.expand_dims(exist_nodes, axis=0)
        cmp_mask = jnp.logical_and(cmp_mask, idx_mask)
        valid_idx = jnp.where(cmp_mask)
        valid_connections = jnp.stack([exist_nodes[valid_idx[0]], exist_nodes[valid_idx[1]]], axis=1)
        # get all connections
        all_connections = self.genes[1:3, :].T

        # remove existing connections
        max_node_id = jnp.max(self.nodes[0, :]) + 1  # e.g. 201
        def flatten_pairs(arr):
            return arr[:, 0] * max_node_id + arr[:, 1]
        valid_codes = flatten_pairs(valid_connections)  # shape (N,)
        all_codes = flatten_pairs(all_connections)
        # ----- 1‑D set difference -----
        unique_codes = jnp.setdiff1d(valid_codes, all_codes, assume_unique=True)
        # ----- decode back to 2‑D (src, dst) -----
        src = unique_codes // max_node_id
        dst = unique_codes % max_node_id
        valid_connections = jnp.stack([src, dst], axis=1)  # shape (M,2)

        # select random connection
        key, subkey = random.split(key)
        conn_idx = random.choice(subkey, valid_connections.shape[0])
        selected_conn = valid_connections[conn_idx]
        # add selected connection
        new_gene = jnp.array([len(innovation_record), selected_conn[0], selected_conn[1]])
        innovation_record = jnp.concatenate([innovation_record, new_gene[:, None]], axis=1)
        key, subkey = random.split(key)
        new_connection = jnp.array([conn_idx, selected_conn[0], selected_conn[1], random.uniform(subkey), 1])
        self.genes = jnp.concatenate([self.genes, new_connection[:, None]], axis=1)
        return innovation_record

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

    def distance(self, other: 'Individual', p: HyperParams) -> float:
        """
        Return distance between two individuals.
        Given by the formula: delta = (c1*E + c2*D) / N + c3*W_avg
        """
        # Get innovation numbers
        a_innovs = self.genes[0, :]
        b_innovs = other.genes[0, :]

        # Matching genes (common innovation numbers)
        common_innovs = jnp.intersect1d(a_innovs, b_innovs)
        a_only_innovs = jnp.setdiff1d(a_innovs, b_innovs, assume_unique=False)
        b_only_innovs = jnp.setdiff1d(b_innovs, a_innovs, assume_unique=False)

        find_index = lambda array, val: jnp.argmax(array ==val)

        if common_innovs.size > 0:
            a_idx = jax.vmap(lambda x: find_index(a_innovs, x))(common_innovs)
            b_idx = jax.vmap(lambda x: find_index(b_innovs, x))(common_innovs)
            a_w = self.genes[3, a_idx]
            b_w = other.genes[3, b_idx]
            W_avg = jnp.mean(jnp.abs(a_w - b_w))
        else:
            W_avg = 0.0

        # Disjoint + Excess genes
        max_common = jnp.minimum(jnp.max(a_innovs), jnp.max(b_innovs))

        # excess
        a_excess = jnp.sum(a_only_innovs > max_common)
        b_excess = jnp.sum(b_only_innovs > max_common)
        E = a_excess + b_excess

        # disjoint
        D = (a_only_innovs.size + b_only_innovs.size) - E

        # Normalize
        N = jnp.maximum(a_innovs.size, b_innovs.size)
        N = jnp.where(N < 20, 1, N)  # if N<20, then N=1

        dist = (p.c1 * E + p.c2 * D) / N + p.c3 * W_avg

        return dist.item()

    def predict(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """

        :param inputs: a flat array of input values
        :return: a flat array of output values
        """
        adj_matrix = self.adj_matrix
        node_ids = self.nodes[0, :].astype(jnp.int32)
        node_types = self.nodes[1, :].astype(jnp.int32)
        node_activations = self.nodes[2, :].astype(jnp.int32)

        # Initialize node values
        n_max = jnp.max(node_ids) + 1
        result = jnp.zeros(n_max, dtype=jnp.float32)

        # Find input nodes and bias nodes
        input_nodes = node_ids[node_types == 1]
        bias_nodes = node_ids[node_types == 4]

        assert inputs.shape[0] == input_nodes.shape[0], "Input size does not match input nodes"

        # set bias
        result = result.at[bias_nodes].set(1.0)
        # set input
        result = result.at[input_nodes].set(inputs)

        levels = self.get_node_levels()  # node id -> node level
        sorted_idx = jnp.argsort(levels)  # sort by node level

        for idx in sorted_idx:
            if levels[idx] < 0:
                continue
            node_id = node_ids[idx]
            node_type = node_types[idx]
            activation = activation_functions[node_activations[idx]]
            if node_type in (2, 3):  # output or hidden node
                # Get incoming connections
                parents = jnp.where(adj_matrix[:, node_id] != 0)[0]
                weights = adj_matrix[parents, node_id]
                inputs = result[parents]
                # Apply activation function
                result = result.at[node_id].set(activation(jnp.dot(weights, inputs)))

        output_mask = node_ids[node_types == 2]
        output = result[output_mask]
        return output

    def __str__(self):
        return f"Individual(fitness={self.fitness}, rank={self.rank}, generation={self.generation}, specie={self.specie})"

    def to_json(self):
        """
        Convert individual to json
        :return: json string
        """
        return {
            "nodes": self.nodes.tolist(),
            "genes": self.genes.tolist(),
            "fitness": self.fitness,
            "rank": self.rank if hasattr(self, "rank") else -1,
            "generation": self.generation,
            "specie": self.specie if hasattr(self, "specie") else -1,
        }

    @classmethod
    def from_json(cls, json_str):
        """
        Convert json to individual
        :param json_str: json string
        :return: individual
        """
        data = json.loads(json_str)
        nodes = jnp.array(data["nodes"])
        genes = jnp.array(data["genes"])
        fitness = data["fitness"]
        rank = data["rank"]
        generation = data["generation"]
        specie = data["specie"]
        individual = cls(nodes, genes)
        individual.fitness = fitness
        individual.rank = rank
        individual.generation = generation
        individual.specie = specie
        return individual
