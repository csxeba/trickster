import numpy as np
from keras.models import Model


class Policy:

    """Light wrapper for a neural network for inspection inside an environment"""

    def __init__(self, model: Model, actions):
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.possible_actions_onehot = np.eye(self.output_dim)
        self.possible_actions = actions

    @staticmethod
    def preprocess(state):
        return state

    def sample(self, state, reward=None):
        del reward
        probabilities = self.model.predict(self.preprocess(state)[None, ...])[0]
        action = np.random.choice(self.action_indices, p=probabilities, size=1)[0]
        return action
