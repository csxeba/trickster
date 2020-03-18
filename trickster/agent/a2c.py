import tensorflow as tf
import gym

from .policy_gradient import PolicyGradient
from ..model import policy, value
from ..processing.reward_shaping import ValueTarget


class A2C(PolicyGradient):

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = None,
                 normalize_advantages: bool = True,
                 value_target: str = ValueTarget.DISCOUNTED,
                 entropy_beta: float = 0.,
                 memory_buffer_size: int = 10000):

        super().__init__(actor, critic, discount_gamma, gae_lambda, normalize_advantages, value_target,
                         entropy_beta, memory_buffer_size)

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         gae_lambda: float = None,
                         normalize_advantages: bool = True,
                         value_target: str = ValueTarget.DISCOUNTED,
                         entropy_beta: float = 0.,
                         memory_buffer_size: int = 10000):

        print(f" [Trickster] - Building A2C for environment: {env.spec.id}")

        if actor == "default":
            print(" [Trickster] - Building the Actor:")
            actor = policy.factory(env, stochastic=True, squash=True, wide=False,
                                   sigma_mode=policy.SigmaMode.STATE_INDEPENDENT)

        if critic == "default":
            print(" [Trickster] - Building the Critic:")
            critic = value.ValueCritic.factory(env.observation_space, wide=True)

        return cls(actor, critic, discount_gamma, gae_lambda, normalize_advantages, value_target,
                   entropy_beta, memory_buffer_size)
