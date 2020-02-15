import gym
import tensorflow as tf

from . import backbones
from . import heads


class Architecture(tf.keras.Model):

    """Composes a backbone (ie. hidden layers) and a head (stochastic or deterministic)"""

    def __init__(self,
                 backbone_model: tf.keras.Model,
                 head_model: tf.keras.Model):

        super().__init__()
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.num_outputs = head_model.num_outputs

    # @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        x = self.backbone_model(x)
        x = self.head_model(x, training)
        return x


class Policy(Architecture):

    """
    Class which is used to represent policies
    - stochastic/deterministic
    - continuous/discreete
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 stochastic=True,
                 squash_continuous=True,
                 action_scaler=None,
                 sigma_predicted=False):

        self.stochastic = stochastic
        backbone_model = backbones.factory(observation_space, wide=False)
        head_model = heads.factory(action_space, stochastic, squash_continuous, action_scaler, sigma_predicted)
        super().__init__(backbone_model, head_model)

    @tf.function(experimental_relax_shapes=True)
    def get_training_outputs(self, inputs, actions):
        if not self.stochastic:
            raise NotImplementedError("Interface only available for stochastic policies")
        features = self.backbone_model(inputs)
        log_prob, entropy = self.head_model.get_training_outputs(features, actions)
        return log_prob, entropy


class Q(Architecture):

    """
    Q network - used in DQN
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space):

        if not isinstance(action_space, gym.spaces.Discrete):
            raise RuntimeError("Non-critic Q-network for non-categorical action space")
        backbone_model = backbones.factory(observation_space, wide=True)
        head_model = heads.DeterministicContinuous(action_space.n, squash=False)
        super().__init__(backbone_model, head_model)


class QCritic(Architecture):

    """Q network - used as a critic in off-policy algos"""

    def __init__(self, observation_space: gym.spaces.Space):
        backbone_model = backbones.factory(observation_space, wide=True)
        head_model = heads.DeterministicContinuous(1, squash=False)
        super().__init__(backbone_model, head_model)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        state, action = inputs
        features = self.backbone_model(state)
        features = tf.concat([features, action], axis=1)
        output = self.head_model(features)
        return output


class ValueCritic(Architecture):

    """Value network - used as a critic in on-policy algos"""

    def __init__(self, observation_space: gym.spaces.Space):
        backbone_model = backbones.factory(observation_space, wide=True)
        head_model = heads.DeterministicContinuous(1, squash=False)
        super().__init__(backbone_model, head_model)
