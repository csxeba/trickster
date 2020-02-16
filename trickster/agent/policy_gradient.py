import numpy as np
import tensorflow as tf

from ..processing import RewardShaper
from ..utility import numeric_utils
from .abstract import RLAgentBase


class PolicyGradient(RLAgentBase):

    transition_memory_keys = ["state", "action", "log_prob", "reward", "done"]
    training_memory_keys = ["state", "action", "returns", "advantages", "log_prob"]
    critic_history_keys = ["critic_loss", "value"]
    actor_history_keys = ["actor_loss", "actor_utility", "actor_utility_std", "actor_entropy", "actor_kld"]

    def __init__(self,
                 actor: tf.keras.Model = "default",
                 critic: tf.keras.Model = None,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = None,
                 normalize_advantages=True,
                 entropy_beta: float = 0.,
                 memory_buffer_size: int = 10000,
                 update_actor: int = 1,
                 update_critic: int = 1):

        super().__init__(memory_buffer_size, separate_training_memory=True)

        self.actor = actor
        self.critic = critic
        self.beta = entropy_beta
        self.history_keys = self.actor_history_keys.copy()
        if self.critic is not None:
            self.history_keys += self.critic_history_keys.copy()
        if gae_lambda is not None and self.critic is None:
            raise RuntimeError("GAE can only be used if a critic network is available")
        self.reward_shaper = RewardShaper(discount_gamma, gae_lambda)
        self.normalize_advantages = normalize_advantages
        self.do_gae = gae_lambda is not None
        self.update_actor = update_actor
        self.update_critic = update_critic

    def _finalize_trajectory(self, final_state=None):
        data = self.transition_memory.as_dict()
        self.transition_memory.reset()

        if self.do_gae:
            critic_input = data["state"]
            if final_state is not None:
                critic_input = tf.concat([critic_input, final_state[None, ...]], axis=0)
            values = self.critic(critic_input)[..., 0].numpy()  # tangling dimension due to Dense(1)
            if final_state is None:
                values = np.concatenate([values, [0.]], axis=0)
            advantages, returns = self.reward_shaper.compute_gae(data["reward"], values[:-1], values[1:], data["done"])
        else:
            returns = self.reward_shaper.discount(data["reward"], data["done"])
            advantages = returns.copy()
        if self.normalize_advantages:
            advantages = numeric_utils.safe_normalize(advantages)

        training_data = dict(state=data["state"], action=data["action"], log_prob=data["log_prob"],
                             returns=returns, advantages=advantages)

        self.training_memory.store(training_data)

    def _set_transition(self, state, reward, done, log_prob, action):
        if self.timestep > 0:
            self.transition.set(reward=reward, done=done)
            self.transition_memory.store(self.transition)
        self.timestep += 1
        if not done:
            self.transition.set(state=state, log_prob=log_prob, action=action)
        else:
            self.timestep = 0
            self.episodes += 1
            self._finalize_trajectory(final_state=state)

    def sample(self, state, reward, done):
        result = self.actor(state[None, ...], training=self.learning)
        if self.learning:
            action = result[0][0].numpy()
            log_prob = result[1][0].numpy()
            self._set_transition(state, reward, done, log_prob, action)
        else:
            action = result[0].numpy()
        return action

    def end_trajectory(self):
        if self.learning:
            self._finalize_trajectory(final_state=None)

    @tf.function(experimental_relax_shapes=True)
    def train_step_critic(self, state: tf.Tensor, target: tf.Tensor):
        with tf.GradientTape() as tape:
            value = self.critic(state)[..., 0]
            loss = tf.reduce_mean(tf.square(value - target))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        return {"critic_loss": loss,
                "value": tf.reduce_mean(value)}

    @tf.function(experimental_relax_shapes=True)
    def train_step_actor(self, state, action, advantages, old_log_prob):

        tf.assert_equal(len(old_log_prob.shape), 1)
        tf.assert_equal(len(advantages.shape), 1)

        with tf.GradientTape() as tape:
            log_prob, entropy = self.actor.get_training_outputs(state, action)
            utilities = -log_prob * advantages
            utility = tf.reduce_mean(utilities)

            entropy = tf.reduce_mean(entropy)

            loss = utility - self.beta * entropy

        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        utility_std = tf.math.reduce_std(utilities)
        kld = tf.reduce_mean(old_log_prob - log_prob)

        return {"actor_loss": loss,
                "actor_utility": utility,
                "actor_utility_std": utility_std,
                "actor_entropy": entropy,
                "actor_kld": kld}

    def fit(self, batch_size=None) -> dict:

        data = self.memory_sampler.sample(size=-1)
        self.memory_sampler.reset()

        data = {k: tf.convert_to_tensor(v, dtype="float32") for k, v in data.items()}

        history = {}
        if self.critic is not None and self.update_critic:
            critic_history = self.train_step_critic(data["state"], data["returns"])
            history.update(critic_history)
        if self.update_actor:
            actor_history = self.train_step_actor(data["state"], data["action"], data["advantages"], data["log_prob"])
            history.update(actor_history)

        return history

    def get_savables(self) -> dict:
        pfx = self.__class__.__name__
        result = {f"{pfx}_actor": self.actor}
        if self.critic is not None:
            result[f"{pfx}_critic"] = self.critic
        return result
