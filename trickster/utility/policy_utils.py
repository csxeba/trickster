import numpy as np
import tensorflow as tf


def sample_action_stochastic_policy(
        model: tf.keras.Model,
        state: np.ndarray,
        possible_action_indices: np.ndarray,
        learning: bool):

    probabilities = model(state)[0].numpy()
    if learning:
        action = np.squeeze(np.random.choice(possible_action_indices, p=probabilities, size=1))
    else:
        action = np.squeeze(np.argmax(probabilities, axis=-1))
    return action
