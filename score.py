import math
import gymnasium as gym

import jax.numpy as jnp
from jax import random

from neat import Individual
from policy import BasePolicy
from slimevolleygym import slimevolley_env

def score_batch(
    agents: list[tuple[BasePolicy, BasePolicy]],
    key: jnp.ndarray,
    batch_size: int = 16,
):
    scores = jnp.zeros((len(agents), 2), dtype=jnp.float32)
    # batching agents
    batch_size = min(batch_size, len(agents))
    batches = []
    batch_num = math.ceil(len(agents) / batch_size)
    for i in range(batch_num):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(agents))
        batch = agents[start:end]
        batches.append(batch)

    # score agents in parallel
    for i, batch in enumerate(batches):
        n_agents = len(batch)
        # init envs
        envs = gym.make_vec(
            "SlimeVolley-v0",
            num_envs=n_agents,
            vectorization_mode="async",
            render_mode="state",
            config={
                'survival_reward': False,
            }
        )
        key, subkey = random.split(key)
        obs, _ = envs.reset(seed=random.randint(shape=(1,), key=subkey, minval=0, maxval=10000).item())
        steps = jnp.zeros(n_agents, dtype=jnp.int32)
        left_rewards = jnp.zeros(n_agents, dtype=jnp.float32)
        right_rewards = jnp.zeros(n_agents, dtype=jnp.float32)
        done = jnp.zeros(n_agents, dtype=jnp.bool_)

        while not done.all() and any(steps < 3000):
            obs_left, obs_right = obs["agent_left"]['obs'], obs["agent_right"]['obs']
            action = {
                "agent_left": jnp.stack([batch[k][0](obs_left[k])  for k in range(n_agents)]),
                "agent_right": jnp.stack([batch[k][1](obs_right[k]) for k in range(n_agents)])
            }
            obs, rewards, terminateds, truncateds, _ = envs.step(action)
            left_rewards -= rewards
            right_rewards += rewards
            done = done | terminateds
            steps = steps.at[jnp.bitwise_not(done)].set(steps[jnp.bitwise_not(done)] + 1)
        envs.close()

        start = i * batch_size
        end = min((i + 1) * batch_size, len(agents))
        scores = scores.at[start:end, 0].set(left_rewards + steps * 0.01)
        scores = scores.at[start:end, 1].set(right_rewards + steps * 0.01)

    return scores


def score(
    agents: list[Individual],
    key: jnp.ndarray,
    sample_num: int = 10,
    extra_agents=None,
    score_method: str = "reward",
) -> jnp.ndarray:
    assert len(agents) > 0, "No agents to score."
    assert sample_num > 0, "Sample num must be greater than 0."
    agents = [BasePolicy(agent) for agent in agents]
    agent_num = len(agents)

    if len(agents) < sample_num:
        sample_num = len(agents)

    sample_keys = random.split(key, len(agents))
    batches: list[tuple[int, int]] = []
    key: jnp.ndarray

    for i in range(len(agents)):
        right_ids = random.choice(sample_keys[i], len(agents), shape=(sample_num,), replace=False)
        if_switch = random.uniform(sample_keys[i], shape=(len(agents),), minval=0, maxval=1) < 0.5
        if extra_agents:
            replace_num =random.randint(sample_keys[i], (1, ), minval=0, maxval=min(len(extra_agents), sample_num)).item()
            extra_ids = random.choice(sample_keys[i], len(extra_agents), shape=(replace_num,), replace=False)
            right_ids = right_ids.at[extra_ids].set(extra_ids + agent_num)

        for j in right_ids:
            if if_switch[j]:
                batches.append((int(j), i))
            else:
                batches.append((i, int(j)))

    match_pairs = []
    for i, j in batches:
        if i >= agent_num:
            left = extra_agents[i - agent_num]
        else:
            left = agents[i]
        if j >= agent_num:
            right = extra_agents[j - agent_num]
        else:
            right = agents[j]
        match_pairs.append((left.get_forward_function(), right.get_forward_function()))
    results = score_batch(match_pairs, key=key)

    scores = jnp.zeros((len(agents),), dtype=jnp.float32)
    counts = jnp.zeros_like(scores)

    for k, (i, j) in enumerate(batches):
        left_score, right_score = results[k]
        if i < agent_num:
            scores = scores.at[i].add(left_score)
            counts = counts.at[i].add(1)
        if j < agent_num:
            scores = scores.at[j].add(right_score)
            counts = counts.at[j].add(1)

    # Avoid division by 0
    counts = jnp.where(counts == 0, 1, counts)
    scores = scores / counts

    if score_method == "reward":
        return scores
    elif score_method == "point":
        raise ValueError("point not supported yet")
        # return points
    else:
        raise ValueError("score_method must be 'reward' or 'point'.")
