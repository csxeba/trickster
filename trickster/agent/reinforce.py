import tensorflow as tf
import gym

from .policy_gradient import PolicyGradient
from ..model import arch


class REINFORCE(PolicyGradient):

    def __init__(self,
                 actor: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = None,
                 normalize_advantages: bool = True,
                 entropy_beta: float = 0.,
                 memory_buffer_size: int = 10000):

        super().__init__(actor=actor,
                         critic=None,
                         discount_gamma=discount_gamma,
                         gae_lambda=gae_lambda,
                         normalize_advantages=normalize_advantages,
                         entropy_beta=entropy_beta,
                         memory_buffer_size=memory_buffer_size)

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         normalize_advantages: bool = True,
                         entropy_beta: float = 0.,
                         memory_buffer_size: int = 10000):

        if actor == "default":
            actor = arch.Policy(env.observation_space, env.action_space, stochastic=True, squash_continuous=True)
        return cls(actor=actor,
                   discount_gamma=discount_gamma,
                   gae_lambda=None,
                   normalize_advantages=normalize_advantages,
                   entropy_beta=entropy_beta,
                   memory_buffer_size=memory_buffer_size)
