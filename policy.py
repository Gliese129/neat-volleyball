from abc import ABC
import jax.numpy as jnp
from neat import Individual


class BasePolicy(ABC):
    model: Individual
    score: float = 0.0
    action_threshold = [0.7, 0.7, 0.7]  # threshold for action selection

    def __init__(self, model: Individual):
        self.model = model

    def get_action(self, obs: list[float]) -> list[int]:
        """
        Get the action from the model given the state.
        :param obs: The observation from a specific agent.
        :return: The action to be taken.
        """
        obs = jnp.array(obs)
        action = self.model.predict(obs)
        action = action.tolist()
        return [
            1 if action[i] > self.action_threshold[i] else 0 for i in range(len(action))
        ]

    def get_forward_function(self):
        self.model.express()
        return self.model.compiled_forward

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

