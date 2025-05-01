from abc import ABC
import jax.numpy as jnp
from neat import Individual


class BasePolicy(ABC):
    model: Individual
    score: float = 0.0

    def __init__(self, model: Individual):
        self.model = model

    def get_action(self, obs: list[float]) -> list[float]:
        """
        Get the action from the model given the state.
        :param obs: The observation from a specific agent.
        :return: The action to be taken.
        """
        obs = jnp.array(obs)
        action = self.model.predict(obs)
        return action.tolist()

    def get_forward_function(self):
        self.model.express()
        return self.model.compiled_forward

    def __call__(self, *args, **kwargs):
        return self.get_action(*args, **kwargs)

