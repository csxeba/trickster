import gym
import numpy as np
import tensorflow as tf

from .off_policy import OffPolicy
from ..utility import model_utils, action_utils
from ..model import arch


class DQN(OffPolicy):

    history_keys = ["loss", "Q", "epsilon"]

    def __init__(self,
                 model: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 epsilon: float = 0.99,
                 epsilon_decay: float = 1.,
                 epsilon_min: float = 0.1,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000,
                 target_network: tf.keras.Model = None):

        super().__init__(memory_buffer_size)
        self.model = model
        self.epsilon_greedy = action_utils.EpsilonGreedy(epsilon, epsilon_decay, epsilon_min)
        self.target_network = target_network
        self.gamma = discount_gamma
        self.polyak = polyak_tau
        self.has_target_network = self.target_network is not None
        self.num_actions = int(self.model.num_outputs)

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         model: tf.keras.Model = None,
                         discount_gamma: float = 0.99,
                         epsilon: float = 0.99,
                         epsilon_decay: float = 1.,
                         epsilon_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         use_target_network: bool = True,
                         target_network: tf.keras.Model = None,
                         memory_buffer_size: int = 10000):

        if model is None:
            model = arch.Q(env.observation_space, env.action_space)
        if use_target_network and target_network is None:
            target_network = arch.Q(env.observation_space, env.action_space)
        return cls(model, discount_gamma, epsilon, epsilon_decay, epsilon_min,
                   polyak_tau, memory_buffer_size, target_network)

    def sample(self, state, reward, done):
        state = state.astype("float32")
        Q = self.model(state[None, ...])[0]
        action = self.epsilon_greedy.sample(Q, do_update=False)
        if self.learning:
            self._set_transition(state, action, reward, done)
        else:
            action = np.argmax(Q)
        return action

    def end_trajectory(self):
        if self.learning:
            self.epsilon_greedy.update()

    @tf.function
    def train_step_q(self, state, state_next, action, reward, done):
        if self.has_target_network:
            Q_target = self.target_network(state_next)
        else:
            Q_target = self.model(state_next)
        bellman_target = self.gamma * tf.reduce_max(Q_target, axis=1) * (1 - done) + reward
        canvas = tf.one_hot(action, self.num_actions, dtype=tf.float32)
        inverse_canvas = 1 - canvas
        with tf.GradientTape() as tape:
            Q = self.model(state)
            target = canvas * bellman_target[:, None] + inverse_canvas * Q
            target = tf.stop_gradient(target)
            loss = tf.reduce_mean(tf.square(target - Q))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return {"loss": loss, "Q": tf.reduce_mean(tf.reduce_max(Q, axis=1))}

    def fit(self, batch_size=32):
        data = self.memory_sampler.sample(batch_size)
        data = {k: tf.convert_to_tensor(data[k], dtype="float32") if k != "action" else data[k]
                for k in ["state", "state_next", "action", "reward", "done"]}

        history = self.train_step_q(data["state"], data["state_next"], data["action"], data["reward"], data["done"])
        history["epsilon"] = self.epsilon_greedy.epsilon

        if self.has_target_network:
            self.meld_weights(mix_in_ratio=self.polyak)

        return history

    def push_weights(self):
        self.target_network.set_weights(self.model.get_weights())

    def meld_weights(self, mix_in_ratio=1.):
        if mix_in_ratio == 1.:
            self.push_weights()
            return
        model_utils.meld_weights(self.target_network, self.model, mix_in_ratio)

    def get_savables(self):
        return {"DQN_model": self.model}
