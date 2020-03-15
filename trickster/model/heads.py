import gym
import tensorflow as tf
from tensorflow.keras import layers as tfl


class Head(tf.keras.Model):

    def __init__(self, num_outputs: int, activation: str):
        super().__init__()
        print(f" [Trickster] - Building Head with {num_outputs} output nodes")
        self.layer = tfl.Dense(num_outputs, activation=activation)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, **kwargs):
        action = self.layer(inputs)
        return action


def factory(action_space: gym.spaces.Space, activation: str = "linear"):
    if isinstance(action_space, gym.spaces.Box):
        return Head(action_space.shape[0], activation=activation)
    elif isinstance(action_space, gym.spaces.Discrete):
        return Head(action_space.n, activation=activation)
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")
