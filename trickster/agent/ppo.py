import gym
import tensorflow as tf
import tensorflow_probability as tfp

from ..utility import history
from ..model import arch
from .policy_gradient import PolicyGradient


class PPO(PolicyGradient):

    actor_history_keys = PolicyGradient.actor_history_keys + ["clip_rate"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 update_batch_size=32,
                 discount_gamma=0.99,
                 gae_lambda=0.95,
                 entropy_beta=0.005,
                 clip_epsilon=0.2,
                 target_kl_divergence=0.01,
                 normalize_advantages=True,
                 memory_buffer_size=10000,
                 actor_updates=10,
                 critic_updates=10):

        super().__init__(actor, critic, discount_gamma, gae_lambda, normalize_advantages, entropy_beta,
                         memory_buffer_size)
        self.epsilon = clip_epsilon
        self.target_kl = target_kl_divergence
        self.actor_updates = actor_updates
        self.critic_updates = critic_updates
        self.batch_size = update_batch_size

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic: tf.keras.Model = "default",
                         update_batch_size=32,
                         discount_gamma=0.99,
                         gae_lambda=0.95,
                         normalize_advantages=True,
                         entropy_beta=0.005,
                         clip_epsilon=0.2,
                         target_kl_divergence=0.01,
                         memory_buffer_size=10000,
                         actor_updates=10,
                         critic_updates=10):

        if actor == "default":
            actor = arch.Policy(env.observation_space, env.action_space, stochastic=True, squash_continuous=True)
        if critic == "default":
            critic = arch.ValueCritic(env.observation_space)

        return cls(actor, critic, update_batch_size, discount_gamma, gae_lambda,
                   entropy_beta, clip_epsilon, target_kl_divergence, normalize_advantages, memory_buffer_size,
                   actor_updates, critic_updates)

    def train_step_actor(self, state, action, advantage, old_probabilities):

        selection = tf.cast(advantage > 0, tf.float32)
        min_adv = ((1+self.epsilon) * selection + (1-self.epsilon) * (1-selection)) * advantage

        old_log_prob = tf.math.log(old_probabilities)

        with tf.GradientTape() as tape:
            distribution: tfp.distributions.Distribution = self.actor(state)
            new_log_prob = distribution.log_prob(action)
            ratio = tf.exp(new_log_prob - old_log_prob)
            utilities = -tf.minimum(ratio*advantage, min_adv)
            utility = tf.reduce_mean(utilities)

            entropy = -tf.reduce_mean(new_log_prob)

            loss = utility - entropy * self.beta

        if tf.reduce_any(tf.math.is_nan(loss)):
            raise RuntimeError

        gradients = tape.gradient(loss, self.actor.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

        kld = tf.reduce_mean(old_log_prob - new_log_prob)
        utility_std = tf.math.reduce_std(utilities)
        clip_rate = tf.reduce_mean(tf.cast(-utilities == min_adv, tf.float32))

        return {"actor_loss": loss,
                "actor_utility": utility,
                "actor_utility_std": utility_std,
                "actor_entropy": entropy,
                "actor_kld": kld,
                "clip_rate": clip_rate}

    def fit(self, batch_size=None) -> dict:
        # states, actions, returns, advantages, old_probabilities
        data = self.memory_sampler.sample(size=-1)
        num_samples = self.memory_sampler.N

        self.memory_sampler.reset()

        datasets = {key: tf.data.Dataset.from_tensor_slices(data[key].astype("float32"))
                    for key in self.training_memory_keys}
        local_history = history.History(*self.history_keys)

        critic_ds = tf.data.Dataset.zip((datasets["state"], datasets["returns"]))
        critic_ds.shuffle(num_samples).repeat()
        critic_ds = critic_ds.batch(self.batch_size).prefetch(min(3, self.critic_updates))
        for update, data in enumerate(critic_ds, start=1):
            logs = self.train_step_critic(*data)
            local_history.buffer(**logs)
            if update == self.critic_updates:
                break

        actor_ds = tf.data.Dataset.zip((
            datasets["state"], datasets["action"], datasets["advantages"], datasets["probability"]))
        actor_ds.shuffle(num_samples).repeat()
        actor_ds = actor_ds.batch(self.batch_size).prefetch(min(3, self.actor_updates))
        for update, data in enumerate(actor_ds, start=1):
            logs = self.train_step_actor(*data)
            local_history.buffer(**logs)
            if update == self.actor_updates:
                break
            if logs["actor_kld"] > self.target_kl:
                break

        local_history.push_buffer()
        logs = local_history.reduce()

        self.memory_sampler.reset()

        return logs
