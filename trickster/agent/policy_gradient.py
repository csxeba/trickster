import tensorflow as tf
import tensorflow_probability as tfp
import gym

from ..experience import replay_buffer
from ..processing import RewardShaper
from ..model import arch
from ..utility import numeric_utils
from .abstract import RLAgentBase


class PolicyGradient(RLAgentBase):

    transition_memory_keys = ["state", "action", "probability", "reward", "done"]
    training_memory_keys = ["state", "action", "returns", "advantages", "probability"]
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

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         actor: tf.keras.Model = "default",
                         critic: tf.keras.Model = None,
                         transition_memory: replay_buffer.Experience = None,
                         discount_gamma: float = 0.99,
                         gae_lambda: float = None,
                         normalize_advantages: bool = True,
                         entropy_beta: float = 0.,
                         memory_buffer_size: int = 10000):

        if actor == "default":
            actor = arch.Policy(env.observation_space, env.action_space, stochastic=True, squash_continuous=True)
        if critic == "default":
            critic = arch.ValueCritic(env.observation_space)
        return cls(actor, critic, discount_gamma, gae_lambda, normalize_advantages, entropy_beta, memory_buffer_size)

    def _finalize_trajectory(self, final_state=None):
        data = self.transition_memory.as_dict()
        self.transition_memory.reset()

        if self.do_gae:
            critic_input = tf.concat([data["state"], final_state[None, ...]], axis=0)
            values = self.critic(critic_input)[..., 0].numpy()  # tangling dimension due to Dense(1)
            advantages, returns = self.reward_shaper.compute_gae(data["reward"], values[:-1], values[1:], data["done"])
        else:
            returns = self.reward_shaper.discount(data["reward"], data["done"])
            advantages = returns.copy()
        if self.normalize_advantages:
            advantages = numeric_utils.safe_normalize(advantages)
        training_data = dict(state=data["state"], action=data["action"], probability=data["probability"],
                             returns=returns, advantages=advantages)
        self.training_memory.store(training_data)

    def _set_transition(self, state, reward, done, probability, action):
        if self.timestep > 0:
            self.transition.set(reward=reward, done=done)
            self.transition_memory.store(self.transition)
        self.timestep += 1
        if not done:
            self.transition.set(state=state, probability=probability, action=action)
        else:
            self.timestep = 0
            self.episodes += 1
            self._finalize_trajectory(final_state=state)

    def sample(self, state, reward, done):
        result = self.actor(state[None, ...], training=True)
        if self.learning:
            action = result[0][0, 0]
            probability = result[1][0].numpy()
            self._set_transition(state, reward, done, probability, action)
        else:
            action = result
        return action

    def end_trajectory(self):
        if self.learning:
            if self.do_gae:
                final_state = self.transition_memory.pop()[self.transition_memory_keys[0]]
            else:
                final_state = None
            self._finalize_trajectory(final_state)

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
    def train_step_actor(self, state, action, advantages, old_probabilities):

        with tf.GradientTape() as tape:
            log_probability, entropy = self.actor.get_training_outputs(state, action)
            log_probability = tf.reduce_mean(log_probability)
            entropy = tf.reduce_mean(entropy)
            utilities = -log_probability * advantages
            utility = tf.reduce_mean(utilities)
            loss = utility - self.beta * entropy

        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_weights))

        old_log_probability = tf.math.log(old_probabilities)
        utility_std = tf.math.reduce_std(utilities)
        kld = tf.reduce_mean(old_log_probability - log_probability)

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
            actor_history = self.train_step_actor(data["state"], data["action"], data["advantages"], data["probability"])
            history.update(actor_history)

        return history

    def get_savables(self) -> dict:
        result = {"policy_gradient_actor": self.actor}
        if self.critic is not None:
            result["policy_gradient_critic"] = self.critic
        return result
