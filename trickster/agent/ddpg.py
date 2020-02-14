import gym
import numpy as np
import tensorflow as tf

from . import td3
from ..utility import off_policy_utils


class DDPG(td3.TD3):

    history_keys = ["actor_loss", "action", "target_action", "critic_loss", "Q", "sigma"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 actor_target: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 action_noise_sigma: float = 0.1,
                 action_noise_sigma_decay: float = 1.,
                 action_noise_sigma_min: float = 0.1,
                 action_minima: np.ndarray = None,
                 action_maxima: np.ndarray = None,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(actor=actor, actor_target=actor_target,
                         critic1=critic, critic1_target=critic_target,
                         critic2=None, critic2_target=None,
                         memory_buffer_size=memory_buffer_size,
                         discount_gamma=discount_gamma,
                         polyak_tau=polyak_tau,
                         action_noise_sigma=action_noise_sigma,
                         action_noise_sigma_min=action_noise_sigma_min,
                         action_noise_sigma_decay=action_noise_sigma_decay,
                         action_minima=action_minima,
                         action_maxima=action_maxima,
                         target_action_noise_sigma=0.,
                         target_action_noise_clip=0.,
                         update_actor_every=1)

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic: tf.keras.Model = "default",
                         actor_target: tf.keras.Model = "default",
                         critic_target: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         action_noise_sigma: float = 2.,
                         action_noise_sigma_decay: float = 0.9999,
                         action_noise_sigma_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = 10000):

        action_minima = env.action_space.low
        action_maxima = env.action_space.high

        actor, actor_target, critic, critic_target, _, _ = off_policy_utils.sanitize_models(
            env, actor, actor_target, critic, critic_target, None, None
        )

        assert all(abs(mini) == abs(maxi) for mini, maxi in zip(action_minima, action_maxima))

        return cls(actor, critic, actor_target, critic_target, discount_gamma,
                   action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min,
                   action_minima, action_maxima, polyak_tau, memory_buffer_size)
