import tensorflow as tf
import gym

from .policy_gradient import PolicyGradient
from ..model import arch


class A2C(PolicyGradient):

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = None,
                 normalize_advantages: bool = True,
                 entropy_beta: float = 0.,
                 memory_buffer_size: int = 10000):

        super().__init__(actor, critic, discount_gamma, gae_lambda, normalize_advantages, entropy_beta,
                         memory_buffer_size)

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         gae_lambda: float = None,
                         normalize_advantages: bool = True,
                         entropy_beta: float = 0.,
                         memory_buffer_size: int = 10000):

        if actor == "default":
            actor = arch.Policy(env.observation_space,
                                env.action_space,
                                stochastic=True,
                                squash_continuous=True)
        if critic == "default":
            critic = arch.ValueCritic(env.observation_space)
        return cls(actor, critic, discount_gamma, gae_lambda, normalize_advantages,
                   entropy_beta, memory_buffer_size)
