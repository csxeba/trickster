import gym
import tensorflow as tf

from .dqn import DQN
from ..utility import off_policy_utils


class DoubleDQN(DQN):

    def __init__(self,
                 model: tf.keras.Model,
                 target_network: tf.keras.Model,
                 num_actions: int,
                 discount_gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 1.0,
                 epsilon_min: float = 0.1,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000):

        super().__init__(model, num_actions, discount_gamma, epsilon, epsilon_decay, epsilon_min, polyak_tau,
                         memory_buffer_size, target_network)
        assert self.has_target_network

    # noinspection PyMethodOverriding
    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         model: tf.keras.Model = "default",
                         target_network: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         epsilon: float = 0.1,
                         epsilon_decay: float = 1.0,
                         epsilon_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         memory_buffer_size: int = 10000):

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN variants can only be used with discreete action spaces.")

        model, target_network = off_policy_utils.sanitize_models_discreete(
            env, model, target_network, use_target_network=True
        )

        return cls(model, target_network, env.action_space.n, discount_gamma,
                   epsilon, epsilon_decay, epsilon_min, polyak_tau, memory_buffer_size)

    @tf.function(experimental_relax_shapes=True)
    def update_q(self, state, state_next, action, reward, done):

        Q_next = self.critic(state_next)
        Q_target = self.critic_target(state_next)

        action_indices = tf.stack([tf.range(0, len(action)), action], axis=1)
        target_action_mask = Q_next == tf.reduce_max(Q_next, axis=1, keepdims=True)

        bellman_target = Q_target[target_action_mask] * self.gamma * (1 - done) + reward

        with tf.GradientTape() as tape:
            Q_model = self.critic(state)
            Q = tf.gather_nd(Q_model, action_indices)
            squared_error = tf.square(bellman_target - Q)
            loss = tf.reduce_mean(squared_error)

        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        return {"Q/loss": loss,
                "Q/Q": tf.reduce_mean(Q_model),
                "action/mean": tf.reduce_mean(tf.cast(action, tf.float32)),
                "action/std": tf.math.reduce_std(tf.cast(action, tf.float32)),
                "lr/critic": self.critic.optimizer.learning_rate}
