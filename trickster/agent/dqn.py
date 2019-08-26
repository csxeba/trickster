import numpy as np
from keras.models import Model

from ..abstract import AgentBase
from ..experience import Experience
from ..utility import kerasic


class DQN(AgentBase):

    def __init__(self,
                 model: Model,
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
        if use_target_network:
            self.target_network = kerasic.copy_model(self.model)
        else:
            self.target_network = self.model

    def _maybe_decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def sample(self, state, reward, done, testing_rollout=False):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            Q = self.model.predict(self.preprocess(state)[None, ...])[0]
            action = np.argmax(Q)

        if not testing_rollout:
            self._maybe_decay_epsilon()

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.dones.append(done)
            self.actions.append(action)

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)  # 0..t
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [reward])  # 1..t+1
        F = np.array(self.dones[1:] + [done])

        self._reset_direct_memory()

        self.memory.remember(S, A, R, F)

    def fit(self, updates=1, batch_size=32, verbose=1):
        losses = []
        max_q_predictions = []
        for update in range(1, updates+1):
            S, S_, A, R, F = self.memory_sampler.sample(batch_size)
            bellman_targets = self.target_network.predict(S_).max(axis=1)

            Q = self.model.predict(S)
            max_q_predictions.append(Q.max(axis=1))
            Q[range(len(Q)), A] = bellman_targets * self.gamma + R
            Q[F, A[F]] = R[F]

            loss = self.model.train_on_batch(S, Q)
            losses.append(loss)

        return {"loss": np.mean(losses), "q": np.mean(max_q_predictions)}

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
