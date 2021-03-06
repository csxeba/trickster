import gym
import numpy as np
import tensorflow as tf

from .off_policy import OffPolicy
from ..utility import model_utils, off_policy_utils
from ..processing import action_processing


class DQN(OffPolicy):

    progress_keys = ["Q/loss", "Q/Q", "Q/epsilon", "action/mean", "action/std", "lr/Q"]

    def __init__(self,
                 model: tf.keras.Model,
                 num_actions: int,
                 discount_gamma: float = 0.99,
                 epsilon: float = 0.99,
                 epsilon_decay: float = 1.,
                 epsilon_min: float = 0.1,
                 polyak_tau: float = 0.01,
                 memory_buffer_size: int = 10000,
                 target_network: tf.keras.Model = None,
                 action_space_n: int = None):

        super().__init__(critic=model,
                         critic_target=target_network,
                         memory_buffer_size=memory_buffer_size,
                         discount_gamma=discount_gamma,
                         polyak_tau=polyak_tau)

        self.epsilon_greedy = action_processing.EpsilonGreedy(epsilon, epsilon_decay, epsilon_min)
        self.target_network = target_network
        self.has_target_network = self.target_network is not None
        self.num_actions = num_actions

    @classmethod
    def from_environment(cls,
                         env: gym.Env,
                         model: tf.keras.Model = "default",
                         discount_gamma: float = 0.99,
                         epsilon: float = 0.1,
                         epsilon_decay: float = 1.,
                         epsilon_min: float = 0.1,
                         polyak_tau: float = 0.01,
                         use_target_network: bool = True,
                         target_network: tf.keras.Model = "default",
                         memory_buffer_size: int = 10000):

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN variants can only be used with discreete action spaces.")

        model, target_network = off_policy_utils.sanitize_models_discreete(
            env=env, model=model,
            target_network=target_network,
            use_target_network=use_target_network
        )

        return cls(model=model,
                   num_actions=env.action_space.n,
                   discount_gamma=discount_gamma,
                   epsilon=epsilon,
                   epsilon_decay=epsilon_decay,
                   epsilon_min=epsilon_min,
                   polyak_tau=polyak_tau,
                   memory_buffer_size=memory_buffer_size,
                   target_network=target_network)

    def sample(self, state, reward, done):
        state = state.astype("float32")
        Q = self.critic(state[None, ...])[0]
        if self.learning:
            action = self.epsilon_greedy.sample(Q, do_update=False)
            self._set_transition(state=state, action=action, reward=reward, done=done)
        else:
            action = np.argmax(Q)
        return action

    def end_trajectory(self):
        if self.learning:
            self.epsilon_greedy.update()

    @tf.function
    def update_q(self, state, state_next, action, reward, done):
        if self.has_target_network:
            Q_target = self.target_network(state_next)
        else:
            Q_target = self.critic(state_next)

        bellman_target = self.gamma * tf.reduce_max(Q_target, axis=1) * (1 - done) + reward

        canvas = tf.one_hot(action, self.num_actions, dtype=tf.float32)
        inverse_canvas = 1 - canvas

        with tf.GradientTape() as tape:
            Q = self.critic(state)
            target = canvas * bellman_target[:, None] + inverse_canvas * Q
            target = tf.stop_gradient(target)
            loss = tf.reduce_mean(tf.square(target - Q))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

        return {"Q/loss": loss,
                "Q/Q": tf.reduce_mean(tf.reduce_max(Q, axis=1)),
                "action/mean": tf.reduce_mean(tf.cast(action, tf.float32)),
                "action/std": tf.math.reduce_std(tf.cast(action, tf.float32)),
                "lr/Q": self.critic.optimizer.learning_rate}

    def fit(self, batch_size=32):
        data = self.memory_sampler.sample(batch_size)
        data = {k: tf.convert_to_tensor(data[k], dtype="float32") if k != "action" else data[k]
                for k in ["state", "state_next", "action", "reward", "done"]}

        history = self.update_q(data["state"], data["state_next"], data["action"], data["reward"], data["done"])
        history["Q/epsilon"] = self.epsilon_greedy.epsilon

        if self.has_target_network:
            model_utils.meld_weights(self.target_network, self.critic, mix_in_ratio=self.tau)

        return history
