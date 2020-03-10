from typing import Union

import gym
import numpy as np
import tensorflow as tf

from . import td3
from ..utility import off_policy_utils


class DDPG(td3.TD3):

    history_keys = ["actor/loss", "action/mean", "action/std", "critic/loss", "critic/Q", "actor/sigma"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 actor_target: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 action_noise_sigma: float = 0.1,
                 action_noise_sigma_decay: float = 1.,
                 action_noise_sigma_min: float = 0.1,
                 action_minima: Union[np.ndarray, float] = None,
                 action_maxima: Union[np.ndarray, float] = None,
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
                         action_noise_sigma: float = 0.1,
                         action_noise_sigma_decay: float = 1.0,
                         action_noise_sigma_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = 10000):

        action_minima = env.action_space.low
        action_maxima = env.action_space.high

        actor, actor_target, critic, critic_target, _, _ = off_policy_utils.sanitize_models_continuous(
            env, actor, actor_target, critic, critic_target, None, None
        )

        return cls(actor, critic, actor_target, critic_target, discount_gamma,
                   action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min,
                   action_minima, action_maxima, polyak_tau, memory_buffer_size)

    @tf.function
    def update_critic(self, state, action, reward, done, state_next):
        action_target = self.actor_target(state_next)
        action_target = tf.clip_by_value(action_target, self.action_minima, self.action_maxima)
        Q_target = self.critic_target([state_next, action_target])[..., 0]
        bellman_target = reward + self.gamma * (1. - done) * Q_target
        with tf.GradientTape() as tape:
            Q = self.critic([state, action])[..., 0]
            loss = tf.keras.losses.mean_squared_error(bellman_target, Q)
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        return {"critic/loss": loss, "critic/Q": tf.reduce_mean(Q)}
