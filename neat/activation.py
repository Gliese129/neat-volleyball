import jax
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