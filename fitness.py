from typing import Tuple

from neat.genome import Genome
import random
from slime_volleyball.slimevolley_env import SlimeVolleyEnv
from slime_volleyball.baseline_policy import BaselinePolicy
from neat.activation import softmax
from neat.superparams import action_threshold


max_step = 200

def get_score(left: Genome, right: Genome = None) -> Tuple[float, float]:
    """
    Get the score of the left and right genomes
    :param left:
    :param right:
    :return:
    """

    use_baseline = right is None
    if right is None:
        right = BaselinePolicy()  # defaults to use RNN Baseline for player

    env = SlimeVolleyEnv({"survival_reward": True, "human_actions": False})

    obs, _ = env.reset(seed=random.randint(0, 10000))
    steps = 0
    left_reward, right_reward = 0, 0

    terminateds = truncateds = {"__all__": False}
    while steps <= max_step and not terminateds["__all__"] and not truncateds["__all__"]:
        obs_right, obs_left = obs["agent_right"], obs["agent_left"]

        left_action_predict = left.predict(obs_left['obs'])
        right_action_predict = right.predict(obs_right['obs'])

        # softmax
        left_action_predict = softmax(left_action_predict)
        left_action = [0 for _ in range(left_action_predict.shape[0])]
        for i in range(left_action_predict.shape[0]):
            if left_action[i] > action_threshold:
                left_action[i] = 1

        if use_baseline:
            right_action = right_action_predict
        else:
            right_action_predict = softmax(right_action_predict)
            right_action = [0 for _ in range(right_action_predict.shape[0])]
            for i in range(right_action_predict.shape[0]):
                if right_action[i] > action_threshold:
                    right_action[i] = 1


        actions = {"agent_left": left_action, "agent_right": right_action}
        obs, reward, terminateds, truncateds, _ = env.step(actions)

        left_reward += reward["agent_left"]
        right_reward += reward["agent_right"]
        steps += 1
    env.close()
    # return the score of the right agent
    return left_reward, right_reward
