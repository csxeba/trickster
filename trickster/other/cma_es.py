import numpy as np
from keras.models import Model

from trickster.experience.experience import Experience


class CMA_ES:

    def __init__(self, model: Model, env, actions, memory: Experience=None):
        self.env = env
        self.possible_actions = actions
        self.model = model
        self.total_reward = 0

    def sample(self, state, reward):
        self.total_reward += reward
        probabilities = np.squeeze(self.model.predict(state[None, ...]))
        action = np.random.choice(self.possible_actions, p=probabilities)
        return action

    def push_experience(self, final_state, final_reward):
        self.total_reward += final_reward
