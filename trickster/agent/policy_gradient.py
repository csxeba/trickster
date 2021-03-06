import numpy as np
import tensorflow as tf

from ..processing import RewardShaper
from ..processing.reward_shaping import ValueTarget
from .abstract import RLAgentBase


class PolicyGradient(RLAgentBase):

    transition_memory_keys = ["state", "state_next", "action", "log_prob", "reward", "done"]
    training_memory_keys = ["state", "action", "target_value", "advantages", "log_prob"]
    critic_progress_keys = ["critic/loss", "critic/value"]
    actor_progress_keys = ["actor/loss", "actor/entropy", "actor/kld", "action/mean", "action/std"]

    def __init__(self,
                 actor: tf.keras.Model = "default",
                 critic: tf.keras.Model = None,
                 discount_gamma: float = 0.99,
                 gae_lambda: float = None,
                 normalize_advantages: bool = True,
                 value_target: str = ValueTarget.DISCOUNTED,
                 entropy_beta: float = 0.,
                 memory_buffer_size: int = 10000,
                 update_actor: int = 1,
                 update_critic: int = 1):

        """
        :param actor:
            Stochastic policy, represented by a neural network.
        :param critic:
            Value network, predicting the total expected future return, given the current state.
        :param discount_gamma:
            Reward discount factor, used to compute trajectory returns.
        :param gae_lambda:
            TD-delta discount factor, used to estimate advantages in Generalized Advantage Estimation.
        :param normalize_advantages:
            Whether to normalize advantages prior to updating the actor with them.
        :param value_target:
            Strategy of Value network target calculation.
        :param entropy_beta:
            Coefficient for (approximate) entropy regularization. Positive encourages higher entropy.
        :param memory_buffer_size:
            Max size of the memory buffer. None corresponds to no size limit.
        :param update_actor:
            How many iterations to update the actor on a sigle batch of data.
        :param update_critic:
            How many iterations to update the critic on a sigle batch of data.
        """

        super().__init__(memory_buffer_size, separate_training_memory=True)

        self.actor = actor
        self.critic = critic
        self.beta = entropy_beta
        self.progress_keys = self.actor_progress_keys.copy()
        if self.critic is not None:
            self.progress_keys += self.critic_progress_keys.copy()
        if gae_lambda and self.critic is None:
            raise RuntimeError("GAE can only be used if a critic network is available")
        if not gae_lambda and value_target == ValueTarget.GAE_RETURN:
            value_target = ValueTarget.DISCOUNTED
            print(" [Trickster.PolicyGradient] - Warning, value_target reset to DISOUNTED from GAE_RETURN,"
                  " because gae_lambda was not set.")
        self.reward_shaper = RewardShaper(discount_gamma, gae_lambda, normalize_advantages, value_target)
        self.update_actor = update_actor
        self.update_critic = update_critic

    def sample(self, state, reward, done):
        result = self.actor(tf.convert_to_tensor(state[None, ...]), training=tf.convert_to_tensor(self.learning))
        if self.learning:
            action, log_prob = map(lambda ar: ar[0].numpy(), result)
            if np.issubdtype(action.dtype, np.integer):
                action = np.squeeze(action)
            log_prob = np.squeeze(log_prob)
            self._set_transition(state=state, reward=reward, done=done, action=action, log_prob=log_prob)
        else:
            action = result[0][0].numpy()
            if isinstance(action.dtype, np.integer):
                action = np.squeeze(action)
        return action

    def end_trajectory(self):
        if not self.learning or self.transition_memory.N == 0:
            return

        data = self.transition_memory.as_dict()
        final_state = self.transition.data["state"]

        reward_shaper_kwargs = dict(rewards=data["reward"], dones=data["done"])
        if self.critic is not None:
            critic_input = tf.concat([data["state"], final_state[None, ...]], axis=0)
            values = self.critic(critic_input)[..., 0].numpy()
            reward_shaper_kwargs["values"] = values[:-1]
            reward_shaper_kwargs["values_next"] = values[1:]

        shaped_rewards = self.reward_shaper.shape_rewards(**reward_shaper_kwargs)

        self.transition_memory.reset()

        training_data = dict(state=data["state"], action=data["action"], log_prob=data["log_prob"],
                             target_value=shaped_rewards.target_values,
                             advantages=shaped_rewards.advantages)

        self.training_memory.store(training_data)

    @tf.function(experimental_relax_shapes=True)
    def train_step_critic(self, state: tf.Tensor, target: tf.Tensor):
        with tf.GradientTape() as tape:
            value = self.critic(state)[..., 0]
            loss = tf.reduce_mean(tf.square(value - target))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))
        return {"critic/loss": loss,
                "critic/value": tf.reduce_mean(value),
                "learning_rate/critic": self.critic.optimizer.learning_rate}

    @tf.function(experimental_relax_shapes=True)
    def train_step_actor(self, state, action, advantages, old_log_prob):

        tf.assert_equal(len(old_log_prob.shape), 1)
        tf.assert_equal(len(advantages.shape), 1)

        with tf.GradientTape() as tape:
            log_prob = self.actor.log_prob(state, action)
            utilities = -log_prob * advantages
            utility = tf.reduce_mean(utilities)

            entropy = -tf.reduce_mean(log_prob)

            loss = utility - self.beta * entropy

        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        kld = tf.reduce_mean(old_log_prob - log_prob)

        return {"actor/loss": utility,
                "actor/entropy": entropy,
                "actor/kld": kld,
                "action/mean": tf.reduce_mean(action),
                "action/std": tf.math.reduce_std(action),
                "learning_rate/actor": self.actor.optimizer.learning_rate}

    def fit(self, batch_size=None) -> dict:

        data = self.memory_sampler.sample(size=-1)
        self.memory_sampler.reset()

        data = {k: tf.convert_to_tensor(v, dtype="float32") for k, v in data.items()}

        history = {}
        if self.critic is not None and self.update_critic:
            critic_history = self.train_step_critic(data["state"], data["target_value"])
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
