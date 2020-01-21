import numpy as np
import tensorflow as tf
import gym

from ..experience import replay_buffer
from ..utility import space_utils, advantage_utils
from .abstract import RLAgentBase


class PolicyGradient(RLAgentBase):

    memory_keys = ["state", "action", "reward", "done", "logits"]
    critic_history_keys = ["critic_loss", "value"]
    actor_history_keys = ["actor_loss", "actor_utility", "actor_utility_std", "actor_entropy", "actor_kld",
                          "advantage", "advantage_std"]

    def __init__(self,
                 action_space: gym.spaces.Space,
                 actor: tf.keras.Model,
                 critic: tf.keras.Model = None,
                 memory: replay_buffer.Experience = None,
                 discount_factor_gamma: float = 0.99,
                 gae_lambda: float = None,
                 entropy_penalty_beta: float = 0.):

        if memory is None:
            memory = replay_buffer.Experience(self.memory_keys, max_length=None)
        super().__init__(action_space, memory)
        self.actor = actor
        self.critic = critic
        self.gamma = discount_factor_gamma
        self.lambda_ = gae_lambda
        self.beta = entropy_penalty_beta
        if isinstance(self.action_space, str):
            if self.action_space == space_utils.CONTINUOUS:
                raise NotImplementedError("Continuous action spaces is not yet implemented for policy gradient algorithms")
        self.negative_log_probability_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
            name="PolicyLogProbability")
        self.history_keys = self.actor_history_keys.copy()
        if self.critic is not None:
            self.history_keys += self.critic_history_keys.copy()
        if gae_lambda is not None and self.critic is None:
            raise RuntimeError("GAE can only be used if a critic network is available")
        self.transition = replay_buffer.Transition(keys=self.memory_keys)
        self.timestep = 0

    def sample(self, state, reward, done):
        logits = self.actor(state[None, ...])
        probability = tf.keras.activations.softmax(logits)[0].numpy()
        probability = np.squeeze(probability)
        if self.learning:
            action = np.squeeze(np.random.choice(self.action_space, p=probability))
            if self.timestep > 0:
                self.transition.set(reward=reward, done=done)
                self.memory.store_data(**self.transition.read())
            self.timestep += 1
            if not done:
                self.transition.set(state=state, logits=logits, action=action)
            else:
                self.timestep = 0
        else:
            action = np.squeeze(np.argmax(probability))

        return action

    @tf.function
    def train_step_critic(self, state, target):
        with tf.GradientTape() as tape:
            value = self.critic(state)[..., 0]
            loss = tf.reduce_mean(tf.square(value - target))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        return {"critic_loss": loss,
                "value": tf.reduce_mean(value)}

    @tf.function
    def train_step_actor(self, state, action, advantages, old_probability):

        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages)
        advantages = advantages - adv_mean
        if adv_std > 0:
            advantages = advantages / adv_std

        with tf.GradientTape() as tape:
            logits = self.actor(state)
            log_probability = -self.negative_log_probability_loss(action, logits)
            entropy = -tf.reduce_mean(log_probability)
            utilities = -log_probability * advantages
            utility = tf.reduce_mean(utilities)
            loss = utility - self.beta * entropy

        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        old_log_probability = -self.negative_log_probability_loss(action, old_probability)
        utility_std = tf.math.reduce_std(utilities)
        kld = tf.reduce_mean(old_log_probability - log_probability)

        return {"actor_loss": loss,
                "actor_utility": utility,
                "actor_utility_std": utility_std,
                "actor_entropy": entropy,
                "actor_kld": kld,
                "advantage": adv_mean,
                "advantage_std": adv_std}

    def _finalize_trajectory(self, state, reward, done, final_state):
        if self.lambda_ is None:
            returns = advantage_utils.discount(reward, done, self.gamma).astype("float32")
            advantage = returns
        else:
            value = self.critic(tf.concat([state, final_state[None, ...]], axis=0))[..., 0]
            advantage = advantage_utils.compute_gae(
                reward, value[:-1], value[1:], done, self.gamma, self.lambda_).astype("float32")
            returns = advantage + value[:-1]
        return returns, advantage

    def fit(self, batch_size=None) -> dict:

        state, action, reward, done, logits = self.memory_sampler.sample(size=-1)
        state = state.astype("float32")

        returns, advantage = self._finalize_trajectory(state, reward, done)

        history = {}
        if self.critic is not None:
            critic_history = self.train_step_critic(state, returns)
            history.update(critic_history)
        actor_history = self.train_step_actor(state, action, advantage, logits)
        history.update(actor_history)

        self.memory_sampler.reset()

        return history

    def get_savables(self) -> dict:
        result = {"policy_gradient_actor": self.actor}
        if self.critic is not None:
            result["policy_gradient_critic"] = self.critic
        return result
