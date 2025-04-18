from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union

import jax
import jax.numpy as jnp
from jax import random

from neat import Individual
from policy import BasePolicy
from slimevolleygym.slimevolley_env import SlimeVolleyEnv


def score_one(left_agent: BasePolicy, right_agent: BasePolicy, key: jnp.ndarray):
    env = SlimeVolleyEnv({
        'survival_reward': True,
        'human_actions': False,
    })

    key, subkey = random.split(key)
    obs, _ = env.reset(seed=random.randint(shape=(1, ), key=subkey, minval=0, maxval=10000).item())
    steps = 0
    total_reward = {"agent_right": 0, "agent_left": 0}

    terminateds = truncateds = {"__all__": False}
    while not terminateds["__all__"] and not truncateds["__all__"] and steps < 30000:
        obs_left, obs_right = obs["agent_left"], obs["agent_right"]
        action = {
            "agent_left": left_agent(obs_left),
            "agent_right": right_agent(obs_right),
        }
        obs, reward, terminateds, truncateds, _ = env.step(action)
        total_reward["agent_left"] += reward["agent_left"]
        total_reward["agent_right"] += reward["agent_right"]
        steps += 1

    env.close()
    return total_reward["agent_left"], total_reward["agent_right"]


def score_batch(
    batches: list[tuple[BasePolicy, BasePolicy]],
    keys: jnp.ndarray,
):
    def score_fn(i):
        left_agent, right_agent = batches[i]
        return score_one(left_agent, right_agent, keys[i])
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(score_fn, range(len(batches))))
    return results


def score(
    agents: list[Individual],
    key: jnp.ndarray,
    sample_num: int = 2,
    score_method: Union['fitness', 'point'] = "fitness",
) -> jnp.ndarray:
    assert len(agents) > 0, "No agents to score."
    assert sample_num > 0, "Sample num must be greater than 0."
    agents = [BasePolicy(agent) for agent in agents]

    if len(agents) < sample_num:
        sample_num = len(agents)

    sample_keys = random.split(key, len(agents))
    batches: list[tuple[int, int]] = []
    keys: list[jnp.ndarray] = []

    for i in range(len(agents)):
        right_ids = random.choice(sample_keys[i], len(agents), shape=(sample_num,), replace=False)
        for j in right_ids:
            batches.append((i, int(j)))
            keys.append(random.PRNGKey(i * 1000 + int(j)))  # 随便搞个 key

    match_pairs = [(agents[i], agents[j]) for i, j in batches]
    results = score_batch(match_pairs, jnp.array(keys))

    scores = jnp.zeros((len(agents),), dtype=jnp.float32)
    counts = jnp.zeros_like(scores)
    points = jnp.zeros_like(scores)

    for k, (i, j) in enumerate(batches):
        left_score, right_score = results[k]
        scores = scores.at[i].add(left_score)
        scores = scores.at[j].add(right_score)
        counts = counts.at[i].add(1)
        counts = counts.at[j].add(1)
        # win count
        tmp = 1 if left_score > right_score else -1
        points = points.at[i].add(tmp)
        points = points.at[j].add(-tmp)


    # Avoid division by 0
    counts = jnp.where(counts == 0, 1, counts)
    scores = scores / counts

    if score_method == "fitness":
        return scores
    elif score_method == "point":
        return points
    else:
        raise ValueError("score_method must be 'fitness' or 'point'.")
