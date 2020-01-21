import numpy as np


class EpsilonGreedy:

    def __init__(self,
                 epsilon: float = 1.,
                 epsilon_decay: float = 1.,
                 epsilon_min: float = 0.1):

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def sample(self, Q, update=True):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(Q))
        else:
            action = np.argmax(Q)
        if update:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        return action

    def reset(self):
        self.epsilon = self.initial_epsilon
