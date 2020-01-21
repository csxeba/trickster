import tensorflow as tf
import gym

from .policy_gradient import PolicyGradient


class A2C(PolicyGradient):

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 action_space: gym.spaces.Space,
                 discount_factor_gamma: float=0.99,
                 gae_lambda: float=None,
                 entropy_penalty_coef=0.,
                 copy_actor_models=False):

        super().__init__(action_space, actor, critic, None, discount_factor_gamma, gae_lambda, entropy_penalty_coef)
        self.copy_actor_models = copy_actor_models
