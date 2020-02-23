from typing import Union

import gym
import tensorflow as tf
import tensorflow_probability as tfp

from . import backbones, heads, arch


class DeterministicContinuous(arch.Architecture):

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 squash: bool = True,
                 scaler: float = None,
                 wide: bool = False):

        backbone_model = backbones.factory(observation_space, wide=wide)
        head_model = heads.factory(action_space, "tanh" if squash else "linear")
        super().__init__(backbone_model, head_model)
        self.do_scaling = scaler is not None
        self.scaler = scaler
        self.build(input_shape=(None,) + observation_space.shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        x = super().call(x)
        if self.do_scaling:
            x = x * self.scaler
        return x


class StochasticContinuous(tf.keras.Model):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 squash: bool = True,
                 scaler: float = 1.,
                 wide: bool = False,
                 sigma_predicted: bool = False):

        super().__init__()

        mean_backbone = backbones.factory(observation_space, wide)
        mean_head = heads.factory(action_space, activation="linear")

        self.mean_model = arch.Architecture(mean_backbone, mean_head)

        if sigma_predicted:
            print(" [Trickster] - Sigma is predicted")
            log_sigma_backbone = backbones.factory(observation_space, wide)
            log_sigma_head = heads.factory(action_space, activation="linear")

            self.log_sigma: Union[tf.keras.Model, tf.Variable] = \
                arch.Architecture(log_sigma_backbone, log_sigma_head)
        else:
            print(" [Trickster] - Sigma is optimized directly")
            self.log_sigma: Union[tf.keras.Model, tf.Variable] = \
                tf.Variable(initial_value=tf.ones(action_space.shape[0]))

        if squash:
            self.bijector = tfp.bijectors.Tanh()
            print(" [Trickster] - Creating Bijector")
        else:
            self.bijector = None

        if scaler is None:
            scaler = 1.
        print(" [Trickster] - Creating Scaler")
        self.scaler = tfp.bijectors.Scale(scaler)

        self.sigma_predicted = sigma_predicted
        self.do_squash = squash

        self.build(input_shape=(None,) + observation_space.shape)
        self.optimizer = tf.keras.optimizers.Adam(1e-3)

    @tf.function(experimental_relax_shapes=True)
    def get_sigma(self, x):
        if self.sigma_predicted:
            sigma = tf.exp(self.log_sigma(x))
        else:
            sigma = tf.reshape(tf.repeat(self.log_sigma, len(x)), (len(x), -1))
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


class StochasticDiscreete(arch.Architecture):

    def __init__(self,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Discrete,
                 wide: bool = False):

        backbone_model = backbones.factory(observation_space, wide)
        head_model = heads.factory(action_space, activation="linear")

        super().__init__(backbone_model, head_model)
        self.build(input_shape=(None,) + observation_space.shape)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        logits = super().call(x)

        if training:
            distribution = tfp.distributions.Categorical(logits=logits)
            sample = distribution.sample()
            log_prob = distribution.log_prob(sample)
            return sample, log_prob

        return tf.argmax(logits)

    @tf.function(experimental_relax_shapes=True)
    def log_prob(self, inputs, action):
        logits = super().call(inputs)
        distribution = tfp.distributions.Categorical(logits=logits)
        log_prob = distribution.log_prob(action)
        return log_prob


def factory(env: gym.Env,
            stochastic: bool,
            squash: bool = True,
            wide: bool = False,
            sigma_predicted: bool = True):

    if isinstance(env.action_space, gym.spaces.Box):

        tf.assert_equal(tf.abs(env.action_space.high), tf.abs(env.action_space.low))
        scaler = env.action_space.high

        if stochastic and isinstance(env.action_space, gym.spaces.Box):
            return StochasticContinuous(env.observation_space, env.action_space, squash, scaler, wide, sigma_predicted)
        elif not stochastic and isinstance(env.action_space, gym.spaces.Box):
            return DeterministicContinuous(env.observation_space, env.action_space, squash, scaler, wide)
        else:
            raise NotImplementedError

    elif isinstance(env.action_space, gym.spaces.Discrete):

        if not stochastic:
            raise NotImplementedError
        return StochasticDiscreete(env.observation_space, env.action_space, wide)

    else:
        raise NotImplementedError
