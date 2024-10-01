import random
from typing import Tuple

from neat.activation import softmax
from neat.genome import Genome
from neat.superparams import action_threshold
from slime_volleyball.baseline_policy import BaselinePolicy
from slime_volleyball.slimevolley_env import SlimeVolleyEnv



def process_action(action_predict):
    action = [0 for _ in range(action_predict.shape[0])]
    for i in range(action_predict.shape[0]):
        if action_predict[i] > action_threshold:
            action[i] = 1
    return action


def get_score(left: Genome, right: Genome = None, max_step = 200) -> Tuple[float, float]:
    """
    Get the score of the left and right genomes
    :param left:
    :param right:
    :param max_step:
    :return:
    """

    swap_left = random.random() > 0.5
    use_baseline = right is None
    if right is None:
        right = BaselinePolicy()  # defaults to use RNN Baseline for player

    # swap the left and right if swipe_left is True
    if swap_left:
        left, right = right, left

    env = SlimeVolleyEnv({"survival_reward": True, "human_actions": False})

    obs, _ = env.reset(seed=random.randint(0, 10000))
    steps = 0
    left_reward, right_reward = 0, 0

    left_consistent_action_bonus, right_consistent_action_bonus = 0, 0
    left_movement_bonus, right_movement_bonus = 0, 0
    last_left_action, last_right_action = None, None

    terminateds = truncateds = {"__all__": False}
    while steps <= max_step and not terminateds["__all__"] and not truncateds["__all__"]:
        obs_right, obs_left = obs["agent_right"], obs["agent_left"]

        left_action_predict = left.predict(obs_left['obs'])
        right_action_predict = right.predict(obs_right['obs'])

        # softmax
        right_action = right_action_predict
        if type(right_action_predict) is not list:
            right_action_predict = softmax(right_action_predict)
            right_action = process_action(right_action_predict)

        left_action = left_action_predict
        if type(left_action_predict) is not list:
            left_action_predict = softmax(left_action_predict)
            left_action = process_action(left_action_predict)


        actions = {"agent_left": left_action, "agent_right": right_action}
        obs, reward, terminateds, truncateds, _ = env.step(actions)

        # Bonus for consistent actions (e.g., not standing still or jumping repeatedly)
        if left_action != last_left_action:
            left_consistent_action_bonus += 0.1  # Encourage varied actions
        last_left_action = left_action

        if right_action != last_right_action:
            right_consistent_action_bonus += 0.1
        last_right_action = right_action

        # Bonus for movement (AI should not stay still too long)
        if sum(left_action) > 0:  # AI is moving or jumping
            left_movement_bonus += 0.05  # Reward movement
        if sum(right_action) > 0:
            right_movement_bonus += 0.05

        left_reward += reward["agent_left"]
        right_reward += reward["agent_right"]
        steps += 1

    env.close()

    left_reward += left_consistent_action_bonus + left_movement_bonus
    right_reward += right_consistent_action_bonus + right_movement_bonus

    # return the score of the right agent
    if swap_left:
        return right_reward, left_reward
    return left_reward, right_reward