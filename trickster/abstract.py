from typing import Callable

import numpy as np
import keras

from .experience import Experience, ExperienceSampler
from .utility import kerasic, numeric


class AgentBase:

    history_keys = []

    def __init__(self,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None):

        if isinstance(action_space, int):
            action_space = np.arange(action_space)
        if hasattr(action_space, "n"):
            action_space = np.arange(action_space.n)

        if memory is None:
            memory = Experience()
        self.memory = memory
        self.memory_sampler = ExperienceSampler(self.memory)
        self.action_space = action_space
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
        self.gamma = discount_factor_gamma
        self.learning = True
        self.preprocess = self._preprocess_noop if state_preprocessor is None else state_preprocessor

    def _push_direct_memory(self, reward, done):
        S = np.array(self.states)  # 0..t
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [reward])  # 1..t+1
        F = np.array(self.dones[1:] + [done])

        self._reset_direct_memory()

        self.memory.remember(S, A, R, dones=F)

    def _push_direct_experience(self, state, action, reward, done):
        if self.learning:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)

    def set_learning_mode(self, switch: bool):
        self.learning = switch

    @staticmethod
    def _preprocess_noop(state):
        return state

    def sample(self, state, reward, done):
        raise NotImplementedError

    def push_experience(self, state, reward, done):
        self._push_direct_memory(reward, done)

    def _reset_direct_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []
