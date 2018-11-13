import numpy as np
from keras.models import Model, model_from_json

from ..experience import Experience
from . import DQN


class DoubleDQN(DQN):

    def __init__(self, model: Model, actions, memory: Experience, reward_discount_factor=0.99,
                 epsilon=0.99, epsilon_decay=1., epsilon_min=0.1, state_preprocessor=None):
        super().__init__(model, actions, memory,
                         reward_discount_factor,
                         epsilon, epsilon_decay, epsilon_min,
                         state_preprocessor)
        self.double = model_from_json(model.to_json())  # type: Model

    def push_experience(self, final_state, final_reward, done=True):
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [final_reward])  # 1..t+1
        Q = np.array(self.Q)  # 0..t
        S = np.array(self.states)  # 0..t

        dQ = self.double.predict(S, batch_size=8, verbose=0)

        self.Q = []
        self.states = []
        self.actions = []
        self.rewards = []

        Y = Q.copy()
        Y[range(len(Y)-1), A[:-1]] = dQ[range(len(Y)-1), A[:-1]] * self.gamma + R[:-1]
        if done:
            Y[-1, A[-1]] = final_reward

        self.memory.remember(S, Y)

    def push_weights(self, mix_in_ratio=1.):
        """
        :param mix_in_ratio: mix_in_ratio * new_weights + (1. - mix_in_ratio) * old_weights
        :return:
        """

        if mix_in_ratio >= 1.:
            self.double.set_weights(self.model.get_weights())
            return

        W = []
        diff = 0.
        mix_in_inverse = 1. - mix_in_ratio
        for old, new in zip(self.double.get_weights(), self.model.get_weights()):
            w = mix_in_inverse*old + mix_in_ratio*new
            diff += (np.linalg.norm(old - w) / old.size)
            W.append(w)
        self.model.set_weights(W)
        return diff / len(W)
