from typing import Union

import numpy as np
import gym
import tensorflow as tf

from .off_policy import OffPolicy
from ..utility import off_policy_utils


class SAC(OffPolicy):

    """Soft Actor-Critic"""

    history_keys = ["actor/loss", "actor/entropy", "alpha/alpha", "alpha/loss",
                    "critic/loss1", "critic/loss2", "critic/Q1", "critic/Q2",
                    "action/mean", "action/std"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 critic2: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 critic2_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 entropy_alpha: float = 0.1,
                 entropy_target: float = None,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = int(1e4)):

        super().__init__(actor, None, critic, critic_target, critic2, critic2_target,
                         memory_buffer_size, discount_gamma, polyak_tau)
        self.entropy_target = entropy_target
        self.log_alpha = tf.Variable(tf.math.log(entropy_alpha))
        self.optimize_alpha_switch = float(entropy_target is not None)
        if self.entropy_target is None:
            self.entropy_target = 0.

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic1: tf.keras.Model = "default",
                         critic1_target: tf.keras.Model = "default",
                         critic2: tf.keras.Model = "default",
                         critic2_target: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         entropy_alpha: float = 0.1,
                         entropy_target: Union[float, str] = -2.,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = int(1e4)):

        print(f" [Trickster] - Building SAC for environment: {env.spec.id}")

        actor, _, critic1, critic1_target, critic2, critic2_target = off_policy_utils.sanitize_models_continuous(
            env, actor, None, critic1, critic1_target, critic2, critic2_target,
            stochastic_actor=True, squash_actions=True
        )
        if entropy_target == "default":
            entropy_target = -np.prod(env.action_space.shape)
            print(f" [Trickster] - SAC target entropy set to {entropy_target:.2f}")
        return cls(actor, critic1, critic1_target, critic2, critic2_target, discount_gamma,
                   entropy_alpha, entropy_target, polyak_tau, memory_buffer_size)

    def sample(self, state, reward, done):
        result = self.actor(tf.convert_to_tensor(state[None, ...]),
                            training=tf.convert_to_tensor(self.learning))
        if self.learning:
            action, log_prob = map(lambda ar: ar[0].numpy(), result)
            if np.issubdtype(action.dtype, np.integer):
                action = np.squeeze(action)
            self._set_transition(state=state, reward=reward, done=done, action=action)
        else:
            action = result[0][0].numpy()
            if isinstance(action.dtype, np.integer):
                action = np.squeeze(action)
        return action

    @tf.function
    def update_critic(self, state, action, reward, done, state_next):

        # Obtain stochastic actions
        action_target, log_prob = self.actor(state_next, training=True)

        # Calculate Q target
        Q1_target = self.critic_target([state_next, action_target])[..., 0]
        Q2_target = self.critic2_target([state_next, action_target])[..., 0]
        Q_target = tf.minimum(Q1_target, Q2_target)

        # Bellman target for critic Q-networks
        critic_target = reward + self.gamma * (1 - done) * (Q_target - tf.exp(self.log_alpha) * log_prob)

        with tf.GradientTape() as tape:
            Q1 = self.critic([state, action])[..., 0]
            Q2 = self.critic2([state, action])[..., 0]
            loss1 = tf.reduce_mean(tf.square(Q1 - critic_target))
            loss2 = tf.reduce_mean(tf.square(Q2 - critic_target))
            loss = loss1 + loss2

        grads = tape.gradient(
            loss, self.critic.trainable_weights + self.critic2.trainable_weights)
        self.critic.optimizer.apply_gradients(
            zip(grads, self.critic.trainable_weights + self.critic2.trainable_weights))

        return {"critic/Q1": tf.reduce_mean(Q1), "critic/Q2": tf.reduce_mean(Q2),
                "critic/loss1": loss1, "critic/loss2": loss2,
                "actor/entropy": -tf.reduce_mean(log_prob)}

    @tf.function
    def update_actor(self, state):
        # Get the actual value of the entropy coefficient
        alpha = tf.exp(self.log_alpha)

        with tf.GradientTape() as tape:
            action, log_prob = self.actor(state, training=True)
            Q = tf.minimum(self.critic([state, action]), self.critic2([state, action]))[..., 0]

            # Actor is jointly maximizing Q and per-sample entropy
            policy_loss = tf.reduce_mean(
                tf.stop_gradient(alpha) * log_prob - Q)

            # Dual problem of alpha is optimized. optimize_alpha_switch is 0. if we don't optimize alpha
            alpha_loss = -self.optimize_alpha_switch * self.log_alpha * tf.stop_gradient(
                tf.reduce_mean(log_prob) + self.entropy_target)

            loss = policy_loss + alpha_loss

        # Actor optimizer is used to jointly optimize the actor weights and log_alpha
        grads = tape.gradient(loss, self.actor.trainable_weights + [self.log_alpha])
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights + [self.log_alpha]))

        return {"actor/loss": policy_loss,
                "alpha/alpha": alpha,
                "alpha/loss": alpha_loss,
                "action/mean": tf.reduce_mean(action),
                "action/std": tf.math.reduce_std(action)}

    def fit(self, batch_size=None):
        data = self._get_sample(batch_size)
        history = self.update_critic(data["state"], data["action"], data["reward"], data["done"], data["state_next"])
        actor_history = self.update_actor(data["state"])
        history.update(actor_history)
        self.update_targets()
        return history
