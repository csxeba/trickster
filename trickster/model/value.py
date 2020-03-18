import gym
import tensorflow as tf

from . import backbones, heads, arch


class Q(arch.Architecture):

    """Q network - used in DQN with discreete actions"""

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Space,
                action_space: gym.spaces.Discrete,
                wide=True,
                batch_norm: bool = False,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        if not isinstance(action_space, gym.spaces.Discrete):
            raise RuntimeError("Non-critic Q-network for non-categorical action space")
        backbone_model = backbones.factory(observation_space, wide=wide, batch_norm=batch_norm)
        head_model = heads.Head(action_space.n, activation="linear")
        model = tf.keras.models.Sequential([backbone_model, head_model])
        model.build((None,) + observation_space.shape)
        return cls(model, optimizer)


class QCritic(arch.Architecture):

    """Q network - used as a critic in off-policy algos"""

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Box,
                action_space: gym.spaces.Box,
                wide=False,
                batch_norm=False,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        print(" [Trickster] - Building QCritic")
        state_input = tf.keras.Input(observation_space.shape, name="QCritic_state_input")
        action_input = tf.keras.Input(action_space.shape, name="QCritic_action_input")
        state_backbone = backbones.factory(observation_space, wide=wide, batch_norm=batch_norm)
        action_backbone = backbones.factory(action_space, wide=False, batch_norm=batch_norm)
        head = heads.Head(1, activation="linear")

        state_features = state_backbone(state_input)
        action_features = action_backbone(action_input)
        features = tf.concat([state_features, action_features], axis=1)
        output = head(features)

        model = tf.keras.Model([state_input, action_input], output)
        model.build([(None,) + observation_space.shape, (None,) + action_space.shape])

        return cls(model, optimizer)


class ValueCritic(arch.Architecture):

    """Value network - used as a critic in on-policy algos"""

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Box,
                wide=True,
                batch_norm: bool = False,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        backbone_model = backbones.factory(observation_space, wide=wide, batch_norm=batch_norm)
        head_model = heads.Head(1, activation="linear")

        model = tf.keras.models.Sequential([backbone_model, head_model])
        model.build((None,) + observation_space.shape)

        return cls(model, optimizer)
