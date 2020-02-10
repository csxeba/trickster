import gym
import tensorflow as tf

from . import backbones
from . import heads


class Architecture(tf.keras.Model):

    def __init__(self,
                 backbone_model: tf.keras.Model,
                 head_model: tf.keras.Model):

        super().__init__()
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.num_outputs = head_model.num_outputs

    def call(self, x, training=None, mask=None):
        x = self.backbone_model(x)
        x = self.head_model(x)
        return x


class Policy(Architecture):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 stochastic=True,
                 squash_continuous=True,
                 action_scaler=None):

        backbone_model = backbones.factory(observation_space)
        head_model = heads.factory(action_space, stochastic, squash_continuous, action_scaler)
        super().__init__(backbone_model, head_model)


class Q(Architecture):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space):

        if not isinstance(action_space, gym.spaces.Discrete):
            raise RuntimeError("Non-critic Q-network for non-categorical action space")
        backbone_model = backbones.factory(observation_space)
        head_model = heads.DeterministicContinuous(action_space.n, squash=False)
        super().__init__(backbone_model, head_model)


class QCritic(Architecture):

    def __init__(self, observation_space: gym.spaces.Space):
        backbone_model = backbones.factory(observation_space)
        head_model = heads.DeterministicContinuous(1, squash=False)
        super().__init__(backbone_model, head_model)

    def call(self, inputs, training=None, mask=None):
        state, action = inputs
        features = self.backbone_model(state)
        features = tf.concat([features, action], axis=1)
        output = self.head_model(features)
        return output


class ValueCritic(Architecture):

    def __init__(self, observation_space: gym.spaces.Space):
        backbone_model = backbones.factory(observation_space)
        head_model = heads.DeterministicContinuous(1, squash=False)
        super().__init__(backbone_model, head_model)
