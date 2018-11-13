import numpy as np
from keras.models import Model

from ..abstract import AgentBase
from trickster.experience.experience import Experience


class DQN(AgentBase):

    def __init__(self, model: Model, actions, memory: Experience=None, reward_discount_factor=0.99,
                 epsilon=0.99, epsilon_decay=1., epsilon_min=0.1, state_preprocessor=None):
        super().__init__(actions, memory, reward_discount_factor, state_preprocessor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.eye = np.eye(self.output_dim)
        self.Q = []

    def _maybe_decay_epsilon(self):
        if self.epsilon_decay < 1:
            if self.epsilon < self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = self.epsilon_min

    def sample(self, state, reward):
        self.states.append(state)
        self.rewards.append(reward)
        Q = self.model.predict(self.preprocess(state)[None, ...])[0]
        action = np.argmax(Q) if np.random.random() > self.epsilon else np.random.randint(0, len(Q))
        self._maybe_decay_epsilon()
        self.Q.append(Q)
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
        Y = self.model.predict(S)
        Y[range(len(Y)), A] = self.model.predict(S_).max(axis=1) * self.gamma + R
        Y[F, A[F]] = R[F]
        loss = self.model.train_on_batch(S, Y)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        return {"loss": loss}
