import numpy as np

from .experience import Experience


class AgentBase:

    def __init__(self,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None):

        if memory is None:
            memory = Experience(max_length=1000)
        if isinstance(action_space, int):
            action_space = np.arange(action_space)
        if hasattr(action_space, "n"):
            action_space = np.arange(action_space.n)

        self.memory = memory
        self.possible_actions = action_space
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

    def fit(self, batch_size=32, verbose=1):
        raise NotImplementedError

    def _reset_direct_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
