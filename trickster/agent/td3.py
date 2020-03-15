from typing import Union

import gym
import numpy as np
import tensorflow as tf

from . import off_policy
from ..utility import off_policy_utils
from ..processing import action_processing


class TD3(off_policy.OffPolicy):

    history_keys = ["actor/loss", "action/mean", "action/std",
                    "critic/loss1", "critic/loss2", "critic/Q1", "critic/Q2",
                    "actor/sigma"]

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
                 action_minima: Union[np.ndarray, float] = None,
                 action_maxima: Union[np.ndarray, float] = None,
                 target_action_noise_sigma: float = 0.2,
                 target_action_noise_clip: float = 0.5,
                 update_actor_every: int = 2):

        super().__init__(actor, actor_target, critic1, critic1_target, critic2, critic2_target,
                         memory_buffer_size, discount_gamma, polyak_tau)

        self.action_smoother = action_processing.NumericContinuousActionSmoother(
            action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min, action_minima, action_maxima)
        self.target_noise_sigma = target_action_noise_sigma
        self.target_noise_clip = target_action_noise_clip
        if not isinstance(action_minima, np.ndarray):
            action_minima = np.array([action_minima])
        if not isinstance(action_maxima, np.ndarray):
            action_maxima = np.array([action_maxima])
        self.action_minima = tf.convert_to_tensor(action_minima[None, ...], dtype=tf.float32)
        self.action_maxima = tf.convert_to_tensor(action_maxima[None, ...], dtype=tf.float32)
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
                         action_noise_sigma: float = 0.1,
                         action_noise_sigma_decay: float = 1.,
                         action_noise_sigma_min: float = 0.1,
                         target_action_noise_sigma: float = 0.2,
                         target_action_noise_clip: float = 0.5,
                         update_actor_every: int = 2):

        print(f" [Trickster] - Building TD3 for environment: {env.spec.id}")

        action_minima = env.action_space.low
        action_maxima = env.action_space.high

        actor, actor_target, critic1, critic1_target, critic2, critic2_target = \
            off_policy_utils.sanitize_models_continuous(
                env, actor, actor_target, critic1, critic1_target, critic2, critic2_target, stochastic_actor=False)

        return cls(actor, actor_target, critic1, critic1_target, critic2, critic2_target,
                   discount_gamma, memory_buffer_size, polyak_tau,
                   action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min,
                   action_minima, action_maxima, target_action_noise_sigma, target_action_noise_clip,
                   update_actor_every)

    def sample(self, state, reward, done):
        action = self.actor(state[None, ...])
        action = np.clip(action, self.action_minima, self.action_maxima)[0]
        if self.learning:
            action = self.action_smoother.sample(action, do_update=False)
            self._set_transition(state=state, action=action, reward=reward, done=done)
        return action

    @tf.function
    def update_critic(self, state, action, reward, done, state_next):

        action_target = self.actor_target(state_next)
        if self.target_noise_sigma > 0.:
            action_target = action_processing.add_clipped_noise(
                action_target, self.target_noise_sigma, self.target_noise_clip, self.action_minima, self.action_maxima)

        Q1_t = self.critic_target([state_next, action_target])[..., 0]
        Q2_t = self.critic2_target([state_next, action_target])[..., 0]
        target_Q = tf.minimum(Q1_t, Q2_t)

        bellman_target = reward + self.gamma * target_Q * (1 - done)

        with tf.GradientTape() as tape:
            Q1 = self.critic([state, action])[..., 0]
            Q2 = self.critic2([state, action])[..., 0]
            loss1 = tf.reduce_mean(tf.keras.losses.mean_squared_error(bellman_target, Q1))
            loss2 = tf.reduce_mean(tf.keras.losses.mean_squared_error(bellman_target, Q2))
            loss = loss1 + loss2

        grads = tape.gradient(
            loss, self.critic.trainable_weights + self.critic2.trainable_weights)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_weights + self.critic2.trainable_weights))

        return {"critic/loss1": loss1,
                "critic/Q1": tf.reduce_mean(Q1),
                "critic/loss2": loss2,
                "critic/Q2": tf.reduce_mean(Q2)}

    @tf.function
    def update_actor(self, state):
        with tf.GradientTape() as tape:
            actions = self.actor(state)
            actions = tf.clip_by_value(actions, self.action_minima, self.action_maxima)
            Q = self.critic([state, actions])
            loss = -tf.reduce_mean(Q)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        return {"actor/loss": loss,
                "action/mean": tf.reduce_mean(actions),
                "action/std": tf.math.reduce_std(actions)}

    def fit(self, batch_size=32):

        data = self.memory_sampler.sample(batch_size)
        data = {key: tf.convert_to_tensor(value, dtype=tf.float32) for key, value in data.items()}

        history = self.update_critic(data["state"], data["action"], data["reward"], data["done"], data["state_next"])

        if self.update_step % self.update_actor_every == 0:
            actor_history = self.update_actor(data["state"])
            history.update(actor_history)
            self.action_smoother.update()

        history["actor/sigma"] = self.action_smoother.sigma
        self.update_targets()
        return history
