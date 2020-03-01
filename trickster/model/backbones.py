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

    def __init__(self, hiddens: tuple, activation="relu", batch_norm: bool = True):
        stack = []
        for h in hiddens:
            stack.append(tfl.Dense(h, activation=activation))
            if batch_norm:
                stack.append(tfl.BatchNormalization())
        if batch_norm:
            print(" [Trickster.MLP] - BatchNorm active!")
        super().__init__(stack)


class WideMLP(MLP):

    def __init__(self, activation="tanh", batch_norm: bool = True):
        super().__init__(hiddens=(300, 400), activation=activation, batch_norm=batch_norm)


class SlimMLP(MLP):

    def __init__(self, activation="tanh", batch_norm: bool = True):
        super().__init__(hiddens=(64, 64), activation=activation, batch_norm=batch_norm)


class CNN(_LayerStack):

    def __init__(self,
                 num_blocks: int,
                 block_depth: int,
                 width_base: int,
                 batch_norm: bool):

        hiddens = []
        for block in range(1, num_blocks+1):
            for layer_num in range(1, block_depth+1):
                hiddens.append(
                    tfl.Conv2D(width_base*block, kernel_size=3, strides=1, padding="same", activation="relu")
                )
                if batch_norm:
                    hiddens.append(tfl.BatchNormalization())
            hiddens.append(tfl.MaxPool2D())
        hiddens.append(tfl.GlobalAveragePooling2D())
        super().__init__(hiddens)


class SimpleCNN(CNN):

    def __init__(self, batch_norm: bool = True):
        super().__init__(num_blocks=3, block_depth=1, width_base=8, batch_norm=batch_norm)


def factory(observation_space: gym.spaces.Space, wide=False, activation="tanh", batch_norm: bool = False):

    if len(observation_space.shape) == 3:
        backbone = SimpleCNN(batch_norm)
    elif len(observation_space.shape) == 1:
        if wide:
            backbone = WideMLP(activation, batch_norm)
        else:
            backbone = SlimMLP(activation, batch_norm)
    else:
        raise RuntimeError(f"Weird observation space dimensionality: {observation_space.shape}")

    return backbone
