import select
import sys
import time

import numpy as np

from neat import Individual
from policy import BasePolicy
from slimevolleygym.core import constants
from slimevolleygym.slimevolley_env import SlimeVolleyEnv
from slimevolleygym.slimevolley_boost_env import SlimeVolleyBoostEnv

# from slime_volleyball.baseline_policy import BaselinePolicy

agent_path = './output/best.json'

with open(agent_path, 'r') as f:
    model = Individual.from_json(f.read())
ai_policy = BasePolicy(model)


if __name__ == "__main__":
    """
    Example of how to use Gym env, in single or multiplayer setting

    Humans can override controls:

    left Agent:
    W - Jump
    A - Left
    D - Right

    right Agent:
    Up Arrow, Left Arrow, Right Arrow
    """

    if constants.RENDER_MODE:
        from pyglet.window import key
        from time import sleep

    manualAction = (
        [0, 0, 0]
    )  # forward, backward, jump
    otherManualAction = [0, 0, 0]
    manualMode = False

    # taken from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
    def key_press(k, mod):
        global manualMode, manualAction
        if k == key.LEFT:
            manualAction[0] = 1
        if k == key.RIGHT:
            manualAction[1] = 1
        if k == key.UP:
            manualAction[2] = 1
        if k == key.SPACE and len(manualAction) > 3:
            manualAction[3] = 1
        if k == key.LEFT or k == key.RIGHT or k == key.UP:
            manualMode = True


    def key_release(k, mod):
        global manualMode, manualAction
        if k == key.LEFT:
            manualAction[0] = 0
        if k == key.RIGHT:
            manualAction[1] = 0
        if k == key.UP:
            manualAction[2] = 0
        if k == key.SPACE and len(manualAction) > 3:
            manualAction[3] = 0

    # policy = BaselinePolicy()  # defaults to use RNN Baseline for player

    env = SlimeVolleyEnv(
        {"survival_reward": True, "human_actions": True}
    )

    if constants.RENDER_MODE:
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release

    while True:
        # start the game
        obs, _ = env.reset(seed=np.random.randint(0, 10000))
        steps = 0

        terminateds = truncateds = False
        manualMode = False

        while not terminateds and not truncateds:
            obs_right, obs_left = obs["agent_right"]['obs'], obs["agent_left"]['obs']

            left_action = ai_policy.get_action(obs_left)
            right_action = ai_policy.get_action(obs_right)
            if manualMode:
                right_action = manualAction


            actions = {"agent_left": left_action, "agent_right": right_action}
            obs, reward, terminateds, truncateds, _ = env.step(actions)

            steps += 1

            if constants.RENDER_MODE:
                env.render()
                sleep(0.01)

            # make the game go slower for human players to be fair to humans.
            if manualMode:
                if constants.PIXEL_MODE:
                    sleep(0.01)
                else:
                    sleep(0.02)
        print(f"Game ended after {steps} steps")
        #
        print("Press q to exit, any other key to continue (waiting 1 second): ")
        rlist, _, _ = select.select([sys.stdin], [], [], 1)
        if rlist:
            if_exit = sys.stdin.read(1).strip()
            if if_exit == "q":
                manualMode = False
                break

    env.close()
