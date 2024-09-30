import enum
import random

import jax.numpy as jnp


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (jnp.tanh(x / 2.) + 1)

def tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x)

def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)

def softmax(x: jnp.ndarray) -> jnp.ndarray:
    e_x =  jnp.exp(x)
    return e_x / e_x.sum()

activation_dict = {
    'none': lambda x: x,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}

class ActivationFunction(enum.Enum):
    NONE = 'none'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    RELU = 'relu'

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        func = activation_dict[self.value]
        return func(x)

    @classmethod
    def random(cls):
        items = [item for item in cls if item != cls.NONE]
        item = random.choice(items)
        return item

    @classmethod
    def from_str(cls, name: str):
        return cls._value2member_map_[name]

