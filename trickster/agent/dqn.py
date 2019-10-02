import numpy as np
from tensorflow import keras

from ..abstract import RLAgentBase
from ..experience import Experience
from ..utility import kerasic


class DQN(RLAgentBase):

    history_keys = ["loss", "Qs", "epsilon"]

    def __init__(self,
                 model: keras.Model,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 epsilon=0.99,
                 epsilon_decay=1.,
                 epsilon_min=0.1,
                 state_preprocessor=None,
                 use_target_network=True):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.model = model
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_network = None
        if use_target_network:
            self.target_network = kerasic.copy_model(self.model)

    def _maybe_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def sample(self, state, reward, done):
        if self.learning and np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            self._maybe_decay_epsilon()
        else:
            Q = self.model.predict(self.preprocess(state)[None, ...])[0]
            action = np.argmax(Q)

        self._push_step_to_direct_memory_if_learning(state, action, reward, done)

        return action

    def fit(self, updates=1, batch_size=32, polyak_rate=0.01):
        losses = []
        max_q_predictions = []
        for update in range(1, updates+1):
            S, S_, A, R, F = self.memory_sampler.sample(batch_size)

            m = len(S)

            if self.target_network is None:
                target_Qs = self.model.predict(S_).max(axis=1)
            else:
                target_Qs = self.target_network.predict(S_).max(axis=1)

            bellman_reserve = R + self.gamma * target_Qs

            bellman_targets = self.model.predict(S)
            max_q_predictions.append(bellman_targets.max(axis=1))

            bellman_targets[range(m), A] = bellman_reserve
            bellman_targets[F, A[F]] = R[F]

            loss = self.model.train_on_batch(S, bellman_targets)
            losses.append(loss)

        if self.target_network is not None:
            self.meld_weights(mix_in_ratio=polyak_rate)

        return {"loss": np.mean(losses), "Qs": np.mean(max_q_predictions), "epsilon": self.epsilon}

    def push_weights(self):
        self.target_network.set_weights(self.model.get_weights())

    def meld_weights(self, mix_in_ratio=1.):
        """
        :param mix_in_ratio: mix_in_ratio * new_weights + (1. - mix_in_ratio) * old_weights
        :return:
        """

        if mix_in_ratio == 1.:
            self.push_weights()
            return

        kerasic.meld_weights(self.target_network, self.model, mix_in_ratio)

    def get_savables(self):
        return {"model": self.model}
