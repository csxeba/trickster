from typing import Union

import gym
import numpy as np
import tensorflow as tf

from . import off_policy
from ..utility import off_policy_utils
from ..processing import action_processing


class TD3(off_policy.OffPolicy):

    history_keys = ["actor_loss", "action", "t_action", "critic_loss", "Q", "critic2_loss", "Q2", "sigma"]

    def __init__(self,
                 actor: tf.keras.Model,
                 actor_target: tf.keras.Model,
                 critic1: tf.keras.Model,
                 critic1_target: tf.keras.Model,
                 critic2: Union[tf.keras.Model, None],
                 critic2_target: Union[tf.keras.Model, None],
                 discount_gamma: float = 0.99,
                 memory_buffer_size: int = int(1e4),
                 polyak_tau: float = 0.005,
                 action_noise_sigma: float = 0.1,
                 action_noise_sigma_decay: float = 1.,
                 action_noise_sigma_min: float = 0.1,
                 action_minima: np.ndarray = None,
                 action_maxima: np.ndarray = None,
                 target_action_noise_sigma: float = 0.2,
                 target_action_noise_clip: float = 0.5,
                 update_actor_every: int = 2):

        super().__init__(actor, actor_target, critic1, critic1_target, critic2, critic2_target,
                         memory_buffer_size, discount_gamma, polyak_tau)

        self.action_smoother = action_processing.NumericContinuousActionSmoother(
            action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min, action_minima, action_maxima)
        self.target_noise_sigma = target_action_noise_sigma
        self.target_noise_clip = target_action_noise_clip
        self.action_minima = tf.convert_to_tensor(action_minima[None, ...])
        self.action_maxima = tf.convert_to_tensor(action_maxima[None, ...])
        self.update_actor_every = update_actor_every
        self.td3 = self.critic2 is not None
        self.update_step = 0

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         actor_target: tf.keras.Model = "default",
                         critic1: tf.keras.Model = "default",
                         critic1_target: tf.keras.Model = "default",
                         critic2: tf.keras.Model = "default",
                         critic2_target: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         memory_buffer_size: int = 10000,
                         polyak_tau: float = 0.01,
                         action_noise_sigma: float = 2.,
                         action_noise_sigma_decay: float = 0.9999,
                         action_noise_sigma_min: float = 0.1,
                         target_action_noise_sigma: float = 2.,
                         target_action_noise_clip: float = 1.,
                         update_actor_every: int = 2):

        action_minima = env.action_space.low
        action_maxima = env.action_space.high

        actor, actor_target, critic1, critic1_target, critic2, critic2_target = off_policy_utils.sanitize_models(
            env, actor, actor_target, critic1, critic1_target, critic2, critic2_target
        )

        assert all(abs(mini) == abs(maxi) for mini, maxi in zip(action_minima, action_maxima))

        return cls(actor, actor_target, critic1, critic1_target, critic2, critic2_target,
                   discount_gamma, memory_buffer_size, polyak_tau,
                   action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min,
                   action_minima, action_maxima, target_action_noise_sigma, target_action_noise_clip,
                   update_actor_every)

    def sample(self, state, reward, done):
        action = self.actor(state[None, ...])[0]
        if self.learning:
            action = self.action_smoother.sample(action, do_update=False)
            self._set_transition(state, action, reward, done)
        return action

    @tf.function
    def update_critic(self, state, action, reward, done, state_next):

        action_target = self.actor_target(state_next)

        if self.td3:
            action_target = action_processing.add_clipped_noise(
                action, self.target_noise_sigma, self.target_noise_clip, self.action_minima, self.action_maxima)

        target_Q = self.critic_target([state_next, action_target])[..., 0]
        if self.td3:
            Q2 = self.critic2_target([state_next, action_target])[..., 0]
            target_Q = tf.minimum(target_Q, Q2)
        bellman_target = reward + self.gamma * target_Q * (1 - done)

        history = {"t_action": tf.reduce_mean(action_target)}

        with tf.GradientTape() as tape:
            Q = self.critic([state, action])[..., 0]
            loss = tf.reduce_mean(tf.square(Q - bellman_target))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        history["critic_loss"] = loss
        history["Q"] = Q

        if self.td3:

            with tf.GradientTape() as tape:
                Q2 = self.critic2([state, action])[..., 0]
                loss2 = tf.reduce_mean(tf.square(Q2 - bellman_target))
            grads = tape.gradient(loss2, self.critic2.trainable_weights)
            self.critic2.optimizer.apply_gradients(zip(grads, self.critic2.trainable_weights))

            history["critic2_loss"] = loss2
            history["Q2"] = Q2

        return history

    @tf.function
    def update_actor(self, state):
        with tf.GradientTape() as tape:
            actions = self.actor(state)
            Q = self.critic([state, actions])
            loss = -tf.reduce_mean(Q)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        return {"actor_loss": loss, "action": tf.reduce_mean(actions)}

    def fit(self, batch_size=32):

        data = self.memory_sampler.sample(batch_size)
        data = {key: tf.convert_to_tensor(value, dtype=tf.float32) for key, value in data.items()}

        history = self.update_critic(data["state"], data["action"], data["reward"], data["done"], data["state_next"])

        if self.update_step % self.update_actor_every == 0:
            actor_history = self.update_actor(data["state"])
            history.update(actor_history)
            self.action_smoother.update()

        history["sigma"] = self.action_smoother.sigma
        self.update_targets()
        return history
