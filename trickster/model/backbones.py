"""
This module contains some basic MLP architectures, which are more-or-less standard for
Deep Reinforcement Learning.
"""

import gym
import tensorflow as tf
from tensorflow.keras import layers as tfl


class _LayerStack(tf.keras.Model):

    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, *args, **kwargs):
        for layer in self.stack:
            x = layer(x)
        return x


class MLP(_LayerStack):

    def __init__(self, hiddens: tuple):
        hiddens = [tfl.Dense(h, activation="relu") for h in hiddens]
        super().__init__(hiddens)


class WideMLP(MLP):

    def __init__(self):
        super().__init__(hiddens=(300, 400))


class SlimMLP(MLP):

    def __init__(self):
        super().__init__(hiddens=(64, 64))


class CNN(_LayerStack):

    def __init__(self,
                 num_blocks: int,
                 block_depth: int,
                 width_base: int):

        hiddens = []
        for block in range(1, num_blocks+1):
            for layer_num in range(1, block_depth+1):
                self.hiddens.append(
                    tfl.Conv2D(width_base*block, kernel_size=3, strides=1, padding="same", activation="relu")
                )
            hiddens.append(tfl.MaxPool2D())
        super().__init__(hiddens)


class SimpleCNN(CNN):

    def __init__(self):
        super().__init__(num_blocks=3, block_depth=1, width_base=8)


def factory(observation_space: gym.spaces.Space, wide=False):

    if len(observation_space.shape) == 3:
        backbone = SimpleCNN()
    elif len(observation_space.shape) == 1:
        if wide:
            backbone = WideMLP()
        else:
            backbone = SlimMLP()
    else:
        raise RuntimeError(f"Weird observation space dimensionality: {observation_space.shape}")

    return backbone
