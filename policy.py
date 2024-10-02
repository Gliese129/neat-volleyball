import jax.numpy as jnp

from neat.genome import Genome


class GenomePolicy:
    network: Genome
    input_state: jnp.array
    output_state: jnp.array

    def __init__(self, network: Genome):
        self.network = network

    def _forward(self):
        self.output_state = self.network.predict(self.input_state)

    def _set_input_state(self, obs):
        # obs is: (op is opponent). obs is also from perspective of the agent (x values negated for other agent)
        [x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, op_x, op_y, op_vx, op_vy] = obs
        self.input_state = jnp.array(obs)

    def _get_action(self):
        forward = 0
        backward = 0
        jump = 0
        if self.output_state[0] > 0.75:
            forward = 1
        if self.output_state[1] > 0.75:
            backward = 1
        if self.output_state[2] > 0.75:
            jump = 1
        return [forward, backward, jump]

    def predict(self, obs):
        """take obs, update rnn state, return action"""
        if type(obs) == dict:
            obs = list(obs.values())[0]
        self._set_input_state(obs)
        self._forward()
        return self._get_action()
