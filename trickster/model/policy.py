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


class DeterministicContinuous(arch.Architecture):

    """
    Deterministic policy which outputs an n-element continuous vector as action.
    """

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                squash: bool = True,
                scaling: float = 1.,
                wide: bool = False,
                batch_norm: bool = True,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        backbone_model = backbones.factory(observation_space, wide=wide, batch_norm=batch_norm)
        head_model = heads.factory(action_space, "tanh" if squash else "linear", scaling=scaling)
        model = tf.keras.models.Sequential([backbone_model, head_model])
        model.build(input_shape=(None,) + observation_space.shape)
        return cls(model, optimizer)


class StochasticContinuous(StochasticPolicyBase):

    """
    Stochastic policy with actions following a Diagonal Multivariate Gaussian distribution.
    """

    def __init__(self,
                 mean_model: tf.keras.Model,
                 log_sigma_representation: Union[tf.keras.Model, tf.Variable],
                 bijector: tfp.bijectors.Bijector,
                 optimizer: tf.keras.optimizers.Optimizer = "default"):

        super().__init__()
        self.mean_model = mean_model
        self.log_sigma = log_sigma_representation
        self.bijector = bijector
        if optimizer == "default":
            optimizer = tf.keras.optimizers.Adam(1e-3)
        self.optimizer = optimizer
        self.sigma_predicted = isinstance(log_sigma_representation, tf.keras.Model)

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Box,
                action_space: gym.spaces.Box,
                squash: bool = True,
                scaler: float = 1.,
                wide: bool = False,
                sigma_mode: str = SigmaMode.STATE_DEPENDENT,
                batch_norm: bool = True,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        mean_backbone = backbones.factory(observation_space, wide, batch_norm=batch_norm)
        mean_head = heads.factory(action_space, activation="linear")
        mean_model = tf.keras.models.Sequential([mean_backbone, mean_head])

        if sigma_mode == SigmaMode.STATE_DEPENDENT:
            print(" [Trickster] - Sigma is predicted")
            log_sigma_backbone = backbones.factory(observation_space, wide, batch_norm=batch_norm)
            log_sigma_head = heads.factory(action_space, activation="linear")
            log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.keras.models.Sequential([log_sigma_backbone, log_sigma_head])

        elif sigma_mode == SigmaMode.STATE_INDEPENDENT:
            print(" [Trickster] - Sigma is optimized directly")
            log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.Variable(initial_value=tf.math.log(tf.ones(action_space.shape[0], tf.float32)))

        elif sigma_mode == SigmaMode.FIXED:
            print(" [Trickster] - Sigma is not optimized")
            log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.Variable(initial_value=tf.math.log(tf.ones(action_space.shape[0], tf.float32)), trainable=False)

        else:
            raise NotImplementedError(f"Unknown sigma_mode: {sigma_mode}")

        if squash:
            bijector = tfp.bijectors.Chain([tfp.bijectors.Tanh(), tfp.bijectors.Scale(scaler)])
            print(" [Trickster] - Creating Tanh bijector")
            print(f" [Trickster] - Scaler set to {scaler}")
        else:
            scaler = scaler or 1.
            bijector = tfp.bijectors.Scale(scaler)
            print(f" [Trickster] - Scaler set to {scaler}")

        return cls(mean_model, log_sigma, bijector, optimizer)

    @tf.function(experimental_relax_shapes=True)
    def get_sigma(self, x):
        if self.sigma_predicted:
            sigma = tf.exp(self.log_sigma(x))
        else:
            sigma = tf.repeat(tf.exp(self.log_sigma), len(x))
            sigma = tf.reshape(sigma, (-1, len(x)))
            sigma = tf.transpose(sigma)
        return sigma

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):

        mean = self.mean_model(inputs)
        sigma = self.get_sigma(inputs)
        distribution = tfp.distributions.MultivariateNormalDiag(mean, sigma)
        distribution = self.bijector(distribution)
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        return sample, log_prob

    @tf.function(experimental_relax_shapes=True)
    def log_prob(self, inputs, action):
        mean = self.mean_model(inputs)
        sigma = self.get_sigma(inputs)
        distribution = tfp.distributions.MultivariateNormalDiag(mean, sigma)
        distribution = self.bijector(distribution)
        log_prob = distribution.log_prob(action)
        return log_prob


class StochasticDiscreete(StochasticPolicyBase):

    """
    Stochastic policy with Categorical distribution for actions.
    """

    def __init__(self, model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer = "default"):
        super().__init__()
        self.model = model
        if optimizer == "default":
            optimizer = tf.keras.optimizers.Adam(1e-3)
        self.optimizer = optimizer

    @classmethod
    def factory(cls,
                observation_space: gym.spaces.Box,
                action_space: gym.spaces.Discrete,
                wide: bool = False,
                batch_norm: bool = True,
                optimizer: tf.keras.optimizers.Optimizer = "default"):

        backbone_model = backbones.factory(observation_space, wide, batch_norm=batch_norm)
        head_model = heads.factory(action_space, activation="linear")
        model = tf.keras.models.Sequential([backbone_model, head_model])
        model.build(input_shape=(None,) + observation_space.shape)
        return cls(model, optimizer)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        logits = self.model(x)
        distribution = tfp.distributions.Categorical(logits=logits)
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        return sample, log_prob

    @tf.function(experimental_relax_shapes=True)
    def log_prob(self, inputs, action):
        logits = self.model(inputs)
        distribution = tfp.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(action)
        return log_prob


def factory(env,
            stochastic: bool,
            squash: bool = True,
            wide: bool = False,
            sigma_mode: str = SigmaMode.STATE_DEPENDENT,
            batch_norm: bool = False) -> tf.keras.Model:

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
    :param batch_norm:
        Whether to use batch normalization.
    :return:
        A Neural Network representing a policy.
    """

    if hasattr(env.action_space, "low"):

        tf.assert_equal(tf.abs(env.action_space.high), tf.abs(env.action_space.low))
        scaler = env.action_space.high

        if stochastic and isinstance(env.action_space, gym.spaces.Box):
            return StochasticContinuous.factory(
                env.observation_space, env.action_space, squash, scaler, wide, sigma_mode, batch_norm)
        elif not stochastic and isinstance(env.action_space, gym.spaces.Box):
            return DeterministicContinuous.factory(
                env.observation_space, env.action_space, squash, scaler, wide, batch_norm)
        else:
            raise NotImplementedError

    elif hasattr(env.action_space, "n"):

        if not stochastic:
            raise NotImplementedError
        return StochasticDiscreete.factory(
            env.observation_space, env.action_space, wide, batch_norm=batch_norm)

    else:
        raise NotImplementedError
