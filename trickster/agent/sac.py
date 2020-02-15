import gym
import tensorflow as tf

from .off_policy import OffPolicy
from ..utility import off_policy_utils


class SAC(OffPolicy):

    """Soft Actor-Critic"""

    history_keys = ["actor_loss", "action", "action_s", "Q1_loss", "Q1", "Q2_loss", "Q2", "entropy"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 critic2: tf.keras.Model,
                 critic_target: tf.keras.Model,
                 critic2_target: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 entropy_beta: float = 0.1,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(actor, None, critic, critic_target, critic2, critic2_target,
                         memory_buffer_size, discount_gamma, polyak_tau)
        self.beta = entropy_beta

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic1: tf.keras.Model = "default",
                         critic1_target: tf.keras.Model = "default",
                         critic2: tf.keras.Model = "default",
                         critic2_target: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         entropy_beta: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = int(1e4)):

        actor, _, critic1, critic1_target, critic2, critic2_target = off_policy_utils.sanitize_models_continuous(
            env, actor, None, critic1, critic1_target, critic2, critic2_target, stochastic_actor=True
        )
        return cls(actor, critic1, critic1_target, critic2, critic2_target, discount_gamma, entropy_beta, polyak_tau,
                   memory_buffer_size)

    def sample(self, state, reward, done):
        action = self.actor(state[None, ...])[0]
        if self.learning:
            self._set_transition(state, action, reward, done)
        return action

    # noinspection DuplicatedCode
    @tf.function
    def update_critic(self, state, action, reward, done, state_next):

        # Obtain stochastic actions
        action_target, action_prob = self.actor(state_next, training=True)
        log_prob = tf.math.log(action_prob)

        # Calculate Q target
        Q1_target = self.critic_target([state_next, action_target])[..., 0]
        Q2_target = self.critic2_target([state_next, action_target])[..., 0]
        Q_target = tf.minimum(Q1_target, Q2_target)

        # Bellman target for critic Q-networks
        critic_target = reward + self.gamma * (Q_target * (1 - done) - self.beta * log_prob)

        with tf.GradientTape() as tape:
            Q1 = self.critic([state, action])[..., 0]
            loss1 = tf.reduce_mean(tf.square(Q1 - critic_target))
        grads = tape.gradient(loss1, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            Q2 = self.critic2([state, action])[..., 0]
            loss2 = tf.reduce_mean(tf.square(Q2 - critic_target))
        grads = tape.gradient(loss2, self.critic2.trainable_weights)
        self.critic2.optimizer.apply_gradients(zip(grads, self.critic2.trainable_weights))

        return {"Q1": tf.reduce_mean(Q1), "Q2": tf.reduce_mean(Q2),
                "Q1_loss": loss1, "Q2_loss": loss2,
                "entropy": -tf.reduce_mean(log_prob)}

    @tf.function
    def update_actor(self, state):
        with tf.GradientTape() as tape:
            action, prob = self.actor(state, training=True)
            Q = tf.minimum(self.critic([state, action]), self.critic2([state, action]))
            loss = tf.reduce_mean(tf.math.log(prob) - Q)

        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        return {"actor_loss": tf.reduce_mean(loss),
                "action": tf.reduce_mean(action),
                "action_s": tf.math.reduce_std(action)}

    def fit(self, batch_size=None):
        data = self._get_sample(batch_size)
        history = self.update_critic(data["state"], data["action"], data["reward"], data["done"], data["state_next"])
        actor_history = self.update_actor(data["state"])
        history.update(actor_history)

        self.update_targets()
        return history
