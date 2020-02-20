import gym
import tensorflow as tf

from . import backbones, heads, arch


class Q(arch.Architecture):

    """Q network - used in DQN with discreete actions"""

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Discrete,
                 wide=True):

        if not isinstance(action_space, gym.spaces.Discrete):
            raise RuntimeError("Non-critic Q-network for non-categorical action space")
        backbone_model = backbones.factory(observation_space, wide=wide)
        head_model = heads.Head(action_space.n, activation="linear")
        super().__init__(backbone_model, head_model)


class QCritic(tf.keras.Model):

    """Q network - used as a critic in off-policy algos"""

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, wide=False):
        super().__init__()
        self.state_backbone = backbones.factory(observation_space, wide=wide)
        self.action_backbone = backbones.factory(action_space, wide=wide)
        self.head = heads.Head(1, activation="linear")
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        state, action = inputs
        state_features = self.state_backbone(state)
        action_features = self.action_backbone(action)
        features = tf.concat([state_features, action_features], axis=1)
        output = self.head(features)
        return output


class ValueCritic(arch.Architecture):

    """Value network - used as a critic in on-policy algos"""

    def __init__(self, observation_space: gym.spaces.Box, wide=True):
        backbone_model = backbones.factory(observation_space, wide=wide)
        head_model = heads.Head(1, activation="linear")
        super().__init__(backbone_model, head_model)
