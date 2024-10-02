import random
from typing import Tuple

from neat.genome import Genome
from policy import GenomePolicy
from slime_volleyball.baseline_policy import BaselinePolicy
from slime_volleyball.slimevolley_env import SlimeVolleyEnv
import math




def ball_is_hit(agent_action, agent_obs):
    [agent_x, agent_y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = agent_obs['obs']

    hit_range_x = 0.1
    hit_range_y = 0.1

    # check if is in range and is taking actions
    is_in_x_range = abs(agent_x - ball_x) < hit_range_x
    is_in_y_range = abs(agent_y - ball_y) < hit_range_y

    if is_in_x_range and is_in_y_range and any(agent_action):
        return True
    return False


def calculate_distance_to_ball(agent_obs):
    """
    计算AI与球的距离
    :param agent_obs: AI的观察，包括AI位置和球的位置
    :return: AI与球之间的欧几里得距离
    """
    [agent_x, agent_y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = agent_obs['obs']

    distance = ((agent_x - ball_x) ** 2 + (agent_y - ball_y) ** 2) ** 0.5
    return distance


def calculate_close_ball_bonus_exp(ball_distance, max_bonus=0.01, decay_rate=2.0):
    bonus = max_bonus * math.exp(-decay_rate * ball_distance)
    return bonus



def get_score(left: GenomePolicy | Genome, right: GenomePolicy | Genome = None, max_step = 200) -> Tuple[float, float]:
    """
    Get the score of the left and right genomes
    :param left:
    :param right:
    :param max_step:
    :return:
    """
    # hit_bonus = 0.02
    # distance_bonus = 0.01

    if type(left) == Genome:
        left = GenomePolicy(left)
    if type(right) == Genome:
        right = GenomePolicy(right)


    if right is None:
        right = BaselinePolicy()  # defaults to use RNN Baseline for player

    # swap the left and right if swipe_left is True
    swap_left = random.random()
    if swap_left > 0.5:
        left, right = right, left

    env = SlimeVolleyEnv({"survival_reward": True, "human_actions": False})

    obs, _ = env.reset(seed=random.randint(0, 10000))
    steps = 0
    left_reward, right_reward = 0, 0

    idle_steps = 0
    max_idle_steps = max_step // 4

    terminateds = truncateds = {"__all__": False}
    while steps <= max_step and not terminateds["__all__"] and not truncateds["__all__"]:
        obs_right, obs_left = obs["agent_right"], obs["agent_left"]

        left_action = left.predict(obs_left)
        right_action = right.predict(obs_right)

        actions = {"agent_left": left_action, "agent_right": right_action}
        obs, reward, terminateds, truncateds, _ = env.step(actions)

        # Check for idleness
        if not any(left_action) and not any(right_action):
            idle_steps += 1
        else:
            idle_steps = 0  # Reset if any action is taken

        if idle_steps >= max_idle_steps:
            # Early termination due to idleness
            left_reward -= 0.5  # Penalize for idleness
            right_reward -= 0.5
            break

        # hit ball bonus
        if ball_is_hit(left_action, obs_left):
            left_reward += 0.02
        if ball_is_hit(right_action, obs_right):
            right_action += 0.02

        # distance bonus
        left_ball_distance = calculate_distance_to_ball(obs_left)
        left_reward += calculate_close_ball_bonus_exp(left_ball_distance, max_bonus=0.01)

        right_ball_distance = calculate_distance_to_ball(obs_right)
        right_reward += calculate_close_ball_bonus_exp(right_ball_distance, max_bonus=0.01)

        left_reward += reward["agent_left"]
        right_reward += reward["agent_right"]
        steps += 1

    env.close()

    if steps <= 100:
        left_reward, right_reward = -0.1  # game will stop in 100 steps if both take no actions

    # return the score of the right agent
    if swap_left:
        return right_reward, left_reward
    return left_reward, right_reward