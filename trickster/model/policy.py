from typing import Union

import gym
import tensorflow as tf
import tensorflow_probability as tfp

from . import backbones, heads, arch


class SigmaMode:

    STATE_DEPENDENT = "state_dependent"
    STATE_INDEPENDENT = "state_independent"
    FIXED = "fixed"


class StochasticPolicyBase(tf.keras.Model):

    """
    Base class and interface definition for stochastic policies.
    """

    def call(self,
             state: tf.Tensor,
             training: bool = False,
             *args, **kwargs):

        """
        The call interface returns an action, which might be different depending on
        whether the agent is currently training or not.

        :param state:
            batch of inputs for the policy
        :param training:
            flag indicating whether the policy is training
        :return:
            A noisy or a deterministic action
        """

        raise NotImplementedError

    def log_prob(self,
                 state: tf.Tensor,
                 action: tf.Tensor):
        """
        :param state:
            Batch of inputs for the policy
        :param action:
            Actions taken by this or another policy
        :return:
            Log probabilities for the actions under the current policy
        """
        raise NotImplementedError


class DeterministicContinuous(tf.keras.Model):

    """
    Deterministic policy which outputs an n-element continuous vector as action.
    """

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 squash: bool = True,
                 scaler: float = None,
                 wide: bool = False):

        super().__init__()
        self.backbone_model = backbones.factory(observation_space, wide=wide)
        self.head_model = heads.factory(action_space, "tanh" if squash else "linear")
        self.do_scaling = scaler is not None
        self.scaler = scaler
        self.build(input_shape=(None,) + observation_space.shape)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        x = self.backbone_model(x)
        x = self.head_model(x)
        if self.do_scaling:
            x = x * self.scaler
        return x


class StochasticContinuous(StochasticPolicyBase):

    """
    Stochastic policy with actions following a Diagonal Multivariate Gaussian distribution.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 squash: bool = True,
                 scaler: float = 1.,
                 wide: bool = False,
                 sigma_mode: str = SigmaMode.STATE_DEPENDENT):

        super().__init__()

        mean_backbone = backbones.factory(observation_space, wide)
        mean_head = heads.factory(action_space, activation="linear")

        self.mean_model = arch.Architecture(mean_backbone, mean_head)

        if sigma_mode == SigmaMode.STATE_DEPENDENT:
            print(" [Trickster] - Sigma is predicted")
            log_sigma_backbone = backbones.factory(observation_space, wide)
            log_sigma_head = heads.factory(action_space, activation="linear")

            self.log_sigma: Union[tf.keras.Model, tf.Variable] = \
                arch.Architecture(log_sigma_backbone, log_sigma_head)
        elif sigma_mode == SigmaMode.STATE_INDEPENDENT:
            print(" [Trickster] - Sigma is optimized directly")
            self.log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.Variable(initial_value=tf.ones(action_space.shape[0]))
        elif sigma_mode == SigmaMode.FIXED:
            print(" [Trickster] - Sigma is not optimized")
            self.log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.Variable(initial_value=tf.ones(action_space.shape[0]), trainable=False)
        else:
            raise NotImplementedError(f"Unknown sigma_mode: {sigma_mode}")

        if squash:
            self.bijector = tfp.bijectors.Tanh()
            print(" [Trickster] - Creating Tanh bijector")
        else:
            self.bijector = None

        if scaler is None:
            scaler = 1.
        print(f" [Trickster] - Scaler set to {scaler}")
        self.scaler = tfp.bijectors.Scale(scaler)

        self.sigma_predicted = sigma_mode == SigmaMode.STATE_DEPENDENT
        self.do_squash = squash

        self.build(input_shape=(None,) + observation_space.shape)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(experimental_relax_shapes=True)
    def get_sigma(self, x):
        if self.sigma_predicted:
            sigma = tf.exp(self.log_sigma(x))
        else:
            sigma = tf.stack([tf.exp(self.log_sigma)]*len(x), axis=0)
        return sigma

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):

        mean = self.mean_model(inputs)

        if training:
            sigma = self.get_sigma(inputs)
            distribution = tfp.distributions.MultivariateNormalDiag(mean, sigma)
            if self.do_squash:
                distribution = self.bijector(distribution)
            distribution = self.scaler(distribution)
            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)

            return sample, log_prob

        if self.do_squash:
            mean = self.bijector(mean)

        mean = self.scaler(mean)
        return mean

    @tf.function(experimental_relax_shapes=True)
    def log_prob(self, inputs, action):
        mean = self.mean_model(inputs)
        sigma = self.get_sigma(inputs)
        distribution = tfp.distributions.MultivariateNormalDiag(mean, sigma)
        if self.do_squash:
            distribution = self.bijector(distribution)
        distribution = self.scaler(distribution)
        log_prob = distribution.log_prob(action)
        return log_prob


class StochasticDiscreete(StochasticPolicyBase):

    """
    Stochastic policy with Categorical distribution for actions.
    """

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 wide: bool = False):

        super().__init__()
        self.backbone_model = backbones.factory(observation_space, wide)
        self.head_model = heads.factory(action_space, activation="linear")
        self.build(input_shape=(None,) + observation_space.shape)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        features = self.backbone_model(x)
        logits = self.head_model(features)

        if training:
            distribution = tfp.distributions.Categorical(logits=logits)
            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)
            return sample, log_prob

        return tf.argmax(logits)

    @tf.function(experimental_relax_shapes=True)
    def log_prob(self, inputs, action):
        features = self.backbone_model(inputs)
        logits = self.head_model(features)
        distribution = tfp.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(action)
        return log_prob


def factory(env,
            stochastic: bool,
            squash: bool = True,
            wide: bool = False,
            sigma_mode: str = SigmaMode.STATE_DEPENDENT) -> tf.keras.Model:

    """
    This method generates a small neural network policy for the supplied environment.

    :param env:
        Gym-like environment.
    :param stochastic:
        Whether the policy should be stochastic or deterministic.
    :param squash:
        Whether to the actions between action space's "low" and "high" by a scaled tanh transformation.
    :param wide:
        Whether to return the wider (400-300) or slimmer (64-64) architecture.
    :param sigma_mode:
        How stdev is obtained in StochasticContinuous policies.
        STATE_DEPENDENT: sigma will be predicted by the neural network, along with the mean.
        STATE_INDEPENDENT: sigma is an independent parameter and optimized directly by the policy optimizer.
        FIXED: sigma is not optimized by the optimizer. It can be changed from the outside though.
        These constants are defined under SigmaMode, but lower case strings can also be used.
    :return:
        A Neural Network representing a policy.
    """

    if hasattr(env.action_space, "low"):

        tf.assert_equal(tf.abs(env.action_space.high), tf.abs(env.action_space.low))
        scaler = env.action_space.high

        if stochastic and isinstance(env.action_space, gym.spaces.Box):
            return StochasticContinuous(env.observation_space, env.action_space, squash, scaler, wide, sigma_mode)
        elif not stochastic and isinstance(env.action_space, gym.spaces.Box):
            return DeterministicContinuous(env.observation_space, env.action_space, squash, scaler, wide)
        else:
            raise NotImplementedError

    elif hasattr(env.action_space, "n"):

        if not stochastic:
            raise NotImplementedError
        return StochasticDiscreete(env.observation_space, env.action_space, wide)

    else:
        raise NotImplementedError
