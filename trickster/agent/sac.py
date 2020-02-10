import numpy as np
import tensorflow as tf

from .off_policy import OffPolicy
from ..processing import action_smoothing


class SAC(OffPolicy):

    """Soft Actor-Critic"""
    """Under construction"""

    history_keys = ["critic_loss", "critic2_loss", "actor_loss", "sigma"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 critic2: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 critic2_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 entropy_beta: float = 0.1,
                 action_noise_sigma: float = 2.,
                 action_noise_sigma_decay: float = 0.9999,
                 action_noise_sigma_min: float = 0.1,
                 action_minima: np.ndarray = None,
                 action_maxima: np.ndarray = None,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(actor, None, critic, critic_target, critic2, critic2_target,
                         memory_buffer_size, discount_gamma, polyak_tau)
        self.smoother = action_smoothing.ContinuousActionSmoother(
            action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min, action_minima, action_maxima)
        self.beta = entropy_beta

    def update_critic(self):
        ...

    def update_actor(self):
        ...

    def fit(self, batch_size=None):
        ...

    def get_savables(self) -> dict:
        pass
