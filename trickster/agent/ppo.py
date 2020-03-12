import gym
import tensorflow as tf

from ..utility import history
from ..model import policy, value
from .policy_gradient import PolicyGradient


class PPO(PolicyGradient):

    actor_history_keys = PolicyGradient.actor_history_keys + ["actor/cliprate"]

    def __init__(self,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model,
                 update_batch_size: int = 32,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 entropy_beta: float = 0.0,
                 clip_epsilon: float = 0.2,
                 target_kl_divergence: float = 0.01,
                 normalize_advantages: bool = True,
                 memory_buffer_size: int = 10000,
                 actor_updates: int = 10,
                 critic_updates: int = 10):

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
                         update_batch_size: int = 32,
                         discount_gamma: float = 0.99,
                         gae_lambda: float = 0.97,
                         entropy_beta: float = 0.0,
                         clip_epsilon: float = 0.2,
                         target_kl_divergence: float = 0.01,
                         normalize_advantages: bool = True,
                         memory_buffer_size: int = 10000,
                         actor_updates: int = 10,
                         critic_updates: int = 10):

        if actor == "default":
            actor = policy.factory(env, stochastic=True, squash=True, wide=False,
                                   sigma_mode=policy.SigmaMode.STATE_INDEPENDENT)
        if critic == "default":
            critic = value.ValueCritic(env.observation_space, wide=True)

        return cls(actor, critic, update_batch_size, discount_gamma, gae_lambda,
                   entropy_beta, clip_epsilon, target_kl_divergence, normalize_advantages, memory_buffer_size,
                   actor_updates, critic_updates)

    @tf.function(experimental_relax_shapes=True)
    def train_step_actor(self, state, action, advantage, old_log_prob):

        selection = tf.cast(advantage > 0, tf.float32)
        min_adv = ((1+self.epsilon) * selection + (1-self.epsilon) * (1-selection)) * advantage

        with tf.GradientTape() as tape:
            new_log_prob = self.actor.log_prob(state, action)

            ratio = tf.exp(new_log_prob - old_log_prob)
            utilities = -tf.minimum(ratio*advantage, min_adv)
            utility = tf.reduce_mean(utilities)

            entropy = -tf.reduce_mean(new_log_prob)

            loss = utility - entropy * self.beta

        gradients = tape.gradient(loss, self.actor.trainable_weights,
                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

        kld = tf.reduce_mean(old_log_prob - new_log_prob)
        clip_rate = tf.reduce_mean(tf.cast(-utilities == min_adv, tf.float32))

        return {"actor/loss": utility,
                "actor/entropy": entropy,
                "actor/kld": kld,
                "actor/cliprate": clip_rate,
                "action/mean": tf.reduce_mean(action),
                "action/std": tf.math.reduce_std(action),
                "learning_rate/actor": self.actor.optimizer.learning_rate}

    def fit(self, batch_size=None) -> dict:
        # states, actions, returns, advantages, old_log_prob
        data = self.memory_sampler.sample(size=-1)
        num_samples = self.memory_sampler.N

        self.memory_sampler.reset()

        datasets = {key: tf.data.Dataset.from_tensor_slices(data[key].astype("float32"))
                    for key in self.training_memory_keys}
        local_history = history.History()

        critic_ds = tf.data.Dataset.zip((datasets["state"], datasets["returns"]))
        critic_ds.shuffle(num_samples).repeat()
        critic_ds = critic_ds.batch(self.batch_size).prefetch(min(3, self.critic_updates))
        for update, data in enumerate(critic_ds, start=1):
            logs = self.train_step_critic_monte_carlo(*data)
            local_history.buffer(**logs)
            if update == self.critic_updates:
                break

        actor_ds = tf.data.Dataset.zip((
            datasets["state"], datasets["action"], datasets["advantages"], datasets["log_prob"]))
        actor_ds.shuffle(num_samples).repeat()
        actor_ds = actor_ds.batch(self.batch_size).prefetch(min(3, self.actor_updates))
        for update, data in enumerate(actor_ds, start=1):
            logs = self.train_step_actor(*data)
            local_history.buffer(**logs)
            if update == self.actor_updates:
                break
            if logs["actor/kld"] > self.target_kl:
                break

        local_history.push_buffer()
        logs = local_history.last()

        self.memory_sampler.reset()

        return logs
