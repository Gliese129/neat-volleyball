import gymnasium as gym
import slimevolleygym.slimevolley_env
from time import sleep
from pyglet.window import key

from slimevolleygym.baseline_policy import BaselinePolicy

if __name__ == "__main__":

    manualAction = [0, 0, 0]  # forward, backward, jump
    manualMode = False

    def key_press(k, mod):
        global manualMode, manualAction
        if k == key.LEFT:
            manualAction[0] = 1
        if k == key.RIGHT:
            manualAction[1] = 1
        if k == key.UP:
            manualAction[2] = 1
        if k in [key.LEFT, key.RIGHT, key.UP]:
            manualMode = True

    def key_release(k, mod):
        global manualMode, manualAction
        if k == key.LEFT:
            manualAction[0] = 0
        if k == key.RIGHT:
            manualAction[1] = 0
        if k == key.UP:
            manualAction[2] = 0

    # Create environment with render_mode set to 'human'
    env = gym.make("SlimeVolleyPixel-v0")
    env.metadata['render_modes'] = 'human'

    policy = BaselinePolicy()  # Use default policy based on state

    obs = env.reset()
    env.render()

    # Ensure env.viewer is created before binding keyboard event handlers
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
    else:
        print("Unable to bind keyboard event handlers as env.viewer is not yet created.")

    defaultAction = [0, 0, 0]

    for t in range(10000):
        if manualMode:  # Override with keyboard input
            leftAction = manualAction
        else:
            leftAction = defaultAction
        rightAction = defaultAction
        action = {
            "agent_left": leftAction,
            "agent_right": rightAction,
        }
        obs, reward, done, info, _ = env.step(action)

        state = obs['agent_right']
        defaultAction = policy.predict(state)
        sleep(0.02)
        env.render()
        if done:
            obs = env.reset()
        if (t + 1) % 5000 == 0:
            print(t + 1)

    env.close()
