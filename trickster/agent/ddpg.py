import gym
import numpy as np
import tensorflow as tf

from .off_policy import OffPolicy
from ..utility import action_utils
from ..model import arch


class DDPG(OffPolicy):

    history_keys = ["actor_loss", "action_norm", "target_action", "critic_loss", "Q", "target_Q", "sigma"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 actor_target: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 action_noise_sigma: float = 2.,
                 action_noise_sigma_decay: float = 0.9999,
                 action_noise_sigma_min: float = 0.1,
                 action_minima: np.ndarray = None,
                 action_maxima: np.ndarray = None,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(actor=actor, actor_target=actor_target,
                         critic=critic, critic_target=critic_target,
                         memory_buffer_size=memory_buffer_size,
                         discount_gamma=discount_gamma,
                         polyak_tau=polyak_tau)

        self.action_smoother = action_utils.ContinuousActionSmoother(
            action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min, action_minima, action_maxima)

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = None,
                         critic: tf.keras.Model = None,
                         actor_target: tf.keras.Model = None,
                         critic_target: tf.keras.Model = None,
                         discount_gamma: float = 0.99,
                         action_noise_sigma: float = 2.,
                         action_noise_sigma_decay: float = 0.9999,
                         action_noise_sigma_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = 10000):

        action_minima = env.action_space.low
        action_maxima = env.action_space.high

        assert all(abs(mini) == abs(maxi) for mini, maxi in zip(action_minima, action_maxima))

        if actor is None:
            actor = arch.Policy(env.observation_space, env.action_space,
                                stochastic=False, squash_continuous=True, action_scaler=action_maxima)
        if actor_target is None:
            actor_target = arch.Policy(env.observation_space, env.action_space,
                                       stochastic=False, squash_continuous=True, action_scaler=action_maxima)
        if critic is None:
            critic = arch.QCritic(env.observation_space)
        if critic_target is None:
            critic_target = arch.QCritic(env.observation_space)

        return cls(actor, critic, actor_target, critic_target, discount_gamma,
                   action_noise_sigma, action_noise_sigma_decay, action_noise_sigma_min,
                   action_minima, action_maxima, polyak_tau, memory_buffer_size)

    def sample(self, state, reward, done):
        action = self.actor(state[None, ...])[0]

        if self.learning:
            action = self.action_smoother.sample(action, do_update=False)

        self._set_transition(state, action, reward, done)

        return action

    @tf.function
    def update_critic(self, state, action, reward, done, state_next):
        action_target = self.actor_target(state_next)
        target_Q = self.critic_target([state_next, action_target])[..., 0]
        bellman_target = reward + self.gamma * target_Q * (1 - done)
        with tf.GradientTape() as tape:
            Q = self.critic([state, action])[..., 0]
            loss = tf.reduce_mean(tf.square(Q - bellman_target))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        return {"critic_loss": loss, "Q": tf.reduce_mean(Q), "target_Q": tf.reduce_mean(target_Q),
                "target_action": tf.reduce_mean(tf.linalg.norm(action_target, axis=1))}

    @tf.function
    def update_actor(self, state):
        with tf.GradientTape() as tape:
            actions = self.actor(state)
            Q = self.critic([state, actions])
            loss = -tf.reduce_mean(Q)
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))
        action_norm = tf.reduce_mean(tf.linalg.norm(actions, axis=1))
        return {"actor_loss": loss, "action_norm": action_norm}

    def fit(self, batch_size=32):
        data = self.memory_sampler.sample(batch_size)
        data = {key: tf.convert_to_tensor(value, dtype=tf.float32) for key, value in data.items()}
        history = self.update_critic(data["state"], data["action"], data["reward"], data["done"], data["state_next"])
        actor_history = self.update_actor(data["state"])
        history.update(actor_history)
        history["sigma"] = self.action_smoother.sigma
        self.action_smoother.update()
        self.update_targets()
        return history
