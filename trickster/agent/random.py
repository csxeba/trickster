import numpy as np

from ..abstract import RLAgentBase
from ..utility import spaces


class RandomAgent(RLAgentBase):

    def sample(self, state, reward, done):
        if hasattr(self.action_space, "sample"):
            return self.action_space.sample
        if self.action_space == spaces.CONTINUOUS:
            return np.random.uniform(self.action_space.min(axis=0),
                                     self.action_space.max(axis=1), size=(1,) + self.action_space.shape)

    def push_experience(self, state, reward, done):
        pass
