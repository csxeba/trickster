from collections import defaultdict

import tensorflow as tf

from ..experience import Experience
from ..utility import advantage_utils, keras_utils, policy_utils, history
from .policy_gradient import PolicyGradient


class PPO(PolicyGradient):

    def __init__(self,
                 action_space,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 memory: Experience = None,
                 update_batch_size=32,
                 discount_factor_gamma=0.99,
                 gae_lambda=0.95,
                 entropy_penalty_beta=0.005,
                 ratio_clip_epsilon=0.2,
                 target_kl_divergence=0.01,
                 actor_updates=10,
                 critic_updates=10):

        super().__init__(action_space, actor, critic, memory, discount_factor_gamma, gae_lambda, entropy_penalty_beta)
        self.epsilon = ratio_clip_epsilon
        self.target_kl = target_kl_divergence
        self.actor_updates = actor_updates
        self.critic_updates = critic_updates
        self.batch_size = update_batch_size

    def train_step_actor(self, state, action, advantage, old_logits):
        adv_mean = tf.reduce_mean(advantage)
        adv_std = tf.math.reduce_std(advantage)
        advantages = advantage - adv_mean
        if adv_std > 0:
            advantages = advantages / adv_std
        selection = tf.cast(advantages > 0, tf.float32)
        min_adv = (1+self.epsilon) * advantages * selection + (1-self.epsilon) * advantages * (1-selection)

        old_log_prob = -self.negative_log_probability_loss(action, old_logits)

        with tf.GradientTape() as tape:
            new_logits = self.actor(state)
            new_log_prob = -self.negative_log_probability_loss(action, new_logits)
            ratio = tf.exp(new_log_prob - old_log_prob)
            utilities = -tf.minimum(ratio*advantages, min_adv)
            utility = tf.reduce_mean(utilities)

            entropy = -tf.reduce_mean(new_log_prob)

            loss = -entropy * self.beta + utility

        gradients = tape.gradient(loss, self.actor.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

        kld = tf.reduce_mean(old_log_prob - new_log_prob)
        utility_std = tf.math.reduce_std(utilities)

        return {"actor_loss": loss,
                "actor_utility": utility,
                "actor_utility_std": utility_std,
                "actor_entropy": entropy,
                "actor_kld": kld,
                "advantage": adv_mean,
                "advantage_std": adv_std}

    def fit(self, batch_size=None) -> dict:
        state, action, reward, done, old_logits = self.memory_sampler.sample(size=-1)
        state = state.astype("float32")
        num_samples = len(state)

        returns, advantages = self._finalize_trajectory(state, reward, done, self.memory.final_state)

        datasets = tuple(map(tf.data.Dataset.from_tensor_slices, [state, action, advantages, old_logits, returns]))
        local_history = history.History(*self.history_keys)

        critic_ds = tf.data.Dataset.zip((datasets[0], datasets[-1])).shuffle(num_samples).repeat()
        critic_ds = critic_ds.batch(self.batch_size).prefetch(min(3, self.actor_updates))
        for update, data in enumerate(critic_ds, start=1):
            logs = self.train_step_critic(*data)
            local_history.buffer(**logs)
            if update == self.critic_updates:
                break

        actor_ds = tf.data.Dataset.zip(datasets[:4]).shuffle(num_samples).repeat()
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
