import gym
import numpy as np
import tensorflow as tf

from .dqn import DQN


class DoubleDQN(DQN):

    def __init__(self,
                 model: tf.keras.Model,
                 target_network: tf.keras.Model,
                 discount_gamma: float = 0.99,
                 epsilon: float = 1.,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.1,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(model, discount_gamma, epsilon, epsilon_decay, epsilon_min, polyak_tau,
                         memory_buffer_size, target_network)
        assert self.has_target_network

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         model: tf.keras.Model = None,
                         target_network: tf.keras.Model = None,
                         discount_gamma: float = 0.99,
                         epsilon: float = 1.,
                         epsilon_decay: float = 0.99,
                         epsilon_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = 10000):

        return DQN.from_environment(env, model, discount_gamma, epsilon, epsilon_decay, epsilon_min, polyak_tau,
                                    use_target_network=True,
                                    target_network=target_network,
                                    memory_buffer_size=memory_buffer_size)

    @tf.function
    def train_step_q(self, state, state_next, action, reward, done):

        Q_target = self.target_network(state_next)
        target_action = tf.argmax(Q_target)
        canvas = tf.one_hot(target_action, self.num_actions, dtype=tf.float32)
        inverse_canvas = 1 - canvas

        with tf.GradientTape() as tape:

            Q_model = self.model(state)

            bellman_target = self.gamma * tf.gather(Q_model, target_action) * (1 - done) + reward
            target = canvas * bellman_target[:, None] + inverse_canvas * Q_model
            target = tf.stop_gradient(target)

            loss = tf.reduce_mean(tf.square(target - Q_model))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return {"loss": loss, "Q": tf.reduce_mean(Q_model)}
