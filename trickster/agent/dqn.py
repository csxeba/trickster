import numpy as np
import tensorflow as tf

from .abstract import RLAgentBase
from ..experience import Experience
from ..utility import keras_utils, action_utils


class DQN(RLAgentBase):

    history_keys = ["loss", "Q", "epsilon"]
    memory_keys = ["state", "state_next", "action", "reward", "done"]

    def __init__(self,
                 model: tf.keras.Model,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 epsilon=0.99,
                 epsilon_decay=1.,
                 epsilon_min=0.1,
                 polyak_factor=0.01,
                 use_target_network=True):

        if memory is None:
            memory = Experience(self.memory_keys)
        super().__init__(action_space, memory)
        self.model = model
        self.epsilon_greedy = action_utils.EpsilonGreedy(epsilon, epsilon_decay, epsilon_min)
        self.target_network = self.model
        if use_target_network:
            self.target_network = keras_utils.copy_model(self.model)
        self.gamma = discount_factor_gamma
        self.polyak = polyak_factor
        self.has_target_network = use_target_network
        self.num_actions = len(self.action_space)
        self.previous_state = None

    def sample(self, state, reward, done):
        state = state.astype("float32")
        Q = self.model(state[None, ...])[0]
        if self.learning:
            action = self.epsilon_greedy.sample(Q, update=True)
            if self.previous_state is not None:
                self.memory.store_data(
                    state=self.previous_state, state_next=state, action=action, reward=reward, done=done)
            self.previous_state = state
        else:
            action = np.argmax(Q)
        if done:
            self.previous_state = None
        return action

    @tf.function
    def train_step_q(self, state, state_next, action, reward, done):
        done = tf.cast(done, tf.float32)
        reward = tf.cast(reward, tf.float32)
        state = tf.cast(state, tf.float32)
        state_next = tf.cast(state_next, tf.float32)

        action_mask = tf.one_hot(action, depth=self.num_actions)
        action_mask = tf.stop_gradient(action_mask)

        Q_target = self.model(state_next)
        bellman_reserve = tf.reduce_max(Q_target, axis=1) * self.gamma * (1 - done) + reward
        with tf.GradientTape() as tape:
            Q = self.model(state)
            Q_action = tf.reduce_sum(Q * action_mask, axis=1)
            delta = bellman_reserve - Q_action
            loss = tf.reduce_mean(tf.square(delta))
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss, tf.reduce_mean(Q_action)

    def fit(self, batch_size=32, updates=1):
        S, S_, A, R, F = self.memory_sampler.sample(batch_size)
        loss, Q = self.train_step_q(S, S_, A, R, F)

        if self.has_target_network:
            self.meld_weights(mix_in_ratio=self.polyak)

        return {"loss": loss, "Q": Q, "epsilon": self.epsilon_greedy.epsilon}

    def push_weights(self):
        self.target_network.set_weights(self.model.get_weights())

    def meld_weights(self, mix_in_ratio=1.):

        if mix_in_ratio == 1.:
            self.push_weights()
            return

        keras_utils.meld_weights(self.target_network, self.model, mix_in_ratio)

    def get_savables(self):
        return {"DQN_model": self.model}
