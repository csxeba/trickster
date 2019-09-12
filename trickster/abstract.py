from typing import Callable

import numpy as np
import keras

from .experience import Experience, ExperienceSampler
from .utility import kerasic, numeric


class RLAgentBase:

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

    def _push_direct_memory(self, state, reward, done):
        S = np.array(self.states)  # 0..t
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [reward])  # 1..t+1
        F = np.array(self.dones[1:] + [done])

        self._reset_direct_memory()

        self.memory.remember(S, A, R, dones=F, final_state=state)

    def _push_direct_experience(self, state, action, reward, done):
        if self.learning:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)

    def _reset_direct_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []

    def set_learning_mode(self, switch: bool):
        self.learning = switch

    @staticmethod
    def _preprocess_noop(state):
        return state

    def sample(self, state, reward, done):
        raise NotImplementedError

    def get_savables(self) -> dict:
        raise NotImplementedError

    def save(self, artifatory_root=None, experiment_name=None, environment_name=None, **metadata):
        import os
        save_root = artifatory_root
        if artifatory_root is None:
            save_root = "../artifactory"
        if experiment_name is not None:
            save_root = os.path.join(save_root, experiment_name)
        if environment_name is not None:
            save_root = os.path.join(save_root, environment_name)
        save_root = os.path.join(save_root, self.__class__.__name__)
        for savable_name, savable in self.get_savables().items():
            meta_suffix = "".join("-{}_{}".format(k, v) for k, v in metadata.items())
            save_path = os.path.join(save_root, "{}{}.h5".format(savable_name, meta_suffix))
            savable.save(save_path)

    def load(self, loadables: dict):
        saveables = self.get_savables()
        for key, value in loadables.items():
            if isinstance(value, str):
                saveables[key].load_weights(value)
            else:
                saveables[key].set_weights(value)

    def push_experience(self, state, reward, done):
        self._push_direct_memory(state, reward, done)

    def dispatch_workers(self, n=1):
        return [self] * n

    def create_worker(self, **worker_kwargs):
        return self
