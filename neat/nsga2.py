"""
Nsga2 is a common algorithm used in evolutionary algorithms.
It is a multi-objective optimization algorithm that uses a non-dominated sorting approach to find a set of solutions that are Pareto optimal.
"""
import warnings
from typing import Union, Tuple

import jax.numpy as jnp


def nsga_sort(objs: jnp.ndarray, return_front=False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, list[list[int]]]]:
    """
    Sorts the objectives and fitness values using non-dominated sorting.
    Args:
        objs: The objectives to be sorted. Shape: nIndividuals x nObjectives(2 in this case).
        return_front: Whether to return the front of the sorted objectives.
    Returns:
        The rank of each individual in the population.
        [optional] The fronts of the sorted objectives.
    """
    # Sort the objectives using non-dominated sorting
    fronts = get_fronts(objs)

    # Sort the individuals in each front
    for i in range(len(fronts)):
        front_objs = objs[fronts[i]]
        crowd_dist = get_crowd_dist(front_objs)
        front_rank = jnp.argsort(-crowd_dist) # large crowd distance means more space for the individual
        fronts[i] = [fronts[i][j] for j in front_rank]
    # Convert to rank
    sorted_id = [i for front in fronts for i in front]
    rank = jnp.zeros_like(objs[:, 0])
    rank = rank.at[sorted_id].set(jnp.arange(len(sorted_id)))
    if return_front:
        return rank, fronts
    return rank


def get_fronts(objs: jnp.ndarray) -> list[list[int]]:
    """
    Get the fronts of the objectives.
    Args:
        objs: The objectives to be sorted.
    Returns:
        The fronts of the sorted objectives.
    """
    S = [[] for _ in range(objs.shape[0])]  # S[p] index of individuals dominated by p
    fronts = [[]] # fronts[0] index of individuals in the first front
    n = [0] * objs.shape[0] # n[p] number of individuals that dominate p
    rank = [0] * objs.shape[0] # rank[p] rank of individual p

    def dominates(p, q):
        """
        Check if individual p dominates individual q.
        """
        return jnp.all(objs[p] <= objs[q]) and jnp.any(objs[p] < objs[q])

    for p in range(objs.shape[0]):
        for q in range(objs.shape[0]):
            if dominates(p, q):
                S[p].append(q)
            elif dominates(q, p):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    # assign front indices
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        next_front = list(set(next_front))
        fronts.append(next_front)
    fronts = fronts[:-1] # remove the last empty front
    return fronts


def get_crowd_dist(objs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the crowding distance of each individual
    Args:
        objs: The objectives to be sorted
    Returns:
        The crowding distance of each individual
    """
    crowd_dists = []
    for i in range(objs.shape[1]):
        obj = objs[:, i]
        key = jnp.argsort(obj)

        crowd_dist = jnp.zeros_like(obj).at[(key[0], key[-1])].set(jnp.inf)
        norm = obj[key[-1]] - obj[key[0]]
        if norm > 0:
            dists = (obj[key[2:]] - obj[key[:-2]]) / norm
            crowd_dist = crowd_dist.at[key[1:-1]].set(dists)

        # restore the original order
        dist = jnp.zeros_like(crowd_dist)
        dist = jnp.nan_to_num(dist, nan=jnp.inf)
        dist = dist.at[key].set(crowd_dist)
        crowd_dists.append(dist)
    return jnp.stack(crowd_dists, axis=1).sum(axis=1) # sum the crowd distance of each objective