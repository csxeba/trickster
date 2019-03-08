import numpy as np
from keras.models import Model

from .experience import Experience, ExperienceSampler


class AgentBase:

    def __init__(self,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None):

        if isinstance(action_space, int):
            action_space = np.arange(action_space)
        if hasattr(action_space, "n"):
            action_space = np.arange(action_space.n)

        self.memory = memory
        if self.memory is None:
            self.memory_sampler = None
        else:
            self.memory_sampler = ExperienceSampler(self.memory)
        self.action_space = action_space
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.gamma = discount_factor_gamma
        self.learning = True
        self.preprocess = self._preprocess_noop if state_preprocessor is None else state_preprocessor

    def set_learning_mode(self, switch: bool):
        self.learning = switch

    @staticmethod
    def _preprocess_noop(state):
        return state

    def sample(self, state, reward, done):
        raise NotImplementedError

    def push_experience(self, state, reward, done):
        raise NotImplementedError

    def _reset_direct_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
