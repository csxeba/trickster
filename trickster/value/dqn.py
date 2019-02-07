import numpy as np
from keras.models import Model

from ..abstract import AgentBase
from ..experience import Experience
from ..utility.kerasic import copy_model


class DQN(AgentBase):

    def __init__(self,
                 model: Model,
                 actions,
                 memory: Experience=None,
                 reward_discount_factor=0.99,
                 epsilon=0.99,
                 epsilon_decay=1.,
                 epsilon_min=0.1,
                 state_preprocessor=None,
                 use_target_network=True):

        super().__init__(actions, memory, reward_discount_factor, state_preprocessor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eye = np.eye(self.output_dim)
        if use_target_network:
            self.target_network = copy_model(self.model)
        else:
            self.target_network = self.model

    def _maybe_decay_epsilon(self):
        if self.epsilon_decay < 1:
            if self.epsilon < self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def sample(self, state, reward):
        self.states.append(state)
        self.rewards.append(reward)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.possible_actions)
        else:
            Q = self.model.predict(self.preprocess(state)[None, ...])[0]
            action = np.argmax(Q)
        self._maybe_decay_epsilon()
        self.actions.append(action)
        return action

    def push_experience(self, final_state, final_reward, done=True):
        S = np.array(self.states)  # 0..t
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [final_reward])  # 1..t+1
        F = np.zeros(len(S), dtype=bool)
        F[-1] = done

        self.states = []
        self.actions = []
        self.rewards = []

        self.memory.remember(S, A, R, F)

    def fit(self, batch_size=32, verbose=1):
        S, S_, A, R, F = self.memory.sample(batch_size)
        bellman_targets = self.target_network.predict(S_).max(axis=1)

        Q = self.model.predict(S)
        Q[range(len(Q)), A] = bellman_targets * self.gamma + R
        Q[F, A[F]] = R[F]

        loss = self.model.train_on_batch(S, Q)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        return {"loss": loss}

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

        W = []
        mix_in_inverse = 1. - mix_in_ratio
        for old, new in zip(self.target_network.get_weights(), self.model.get_weights()):
            w = mix_in_inverse*old + mix_in_ratio*new
            W.append(w)
        self.model.set_weights(W)
        self.target_network.set_weights(W)
