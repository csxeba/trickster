import numpy as np
import gym

import trickster


class RLAgentBase:

    memory_keys = []
    history_keys = []

    def __init__(self,
                 action_space: gym.spaces.Space,
                 memory: trickster.experience.Experience):

        if isinstance(action_space, np.ndarray):
            pass
        elif isinstance(action_space, int):
            action_space = np.arange(action_space)
        elif isinstance(action_space, gym.spaces.Discrete):
            action_space = np.arange(action_space.n)
        elif isinstance(action_space, gym.spaces.Box):
            action_space = trickster.utility.space_utils.CONTINUOUS
        elif action_space == trickster.utility.space_utils.CONTINUOUS:
            pass
        else:
            assert False

        memory.initialize(self)
        self.memory = memory
        self.memory_sampler = trickster.experience.ExperienceSampler(self.memory)
        self.action_space = action_space
        self.learning = True

    def _push_step_to_direct_memory_if_learning(self, **kwargs):
        if self.learning:
            self.memory.store_data(**kwargs)

    def set_learning_mode(self, switch: bool):
        self.learning = switch

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

    def dispatch_workers(self, n=1):
        return [self] * n

    def create_worker(self, **worker_kwargs):
        return self

    def fit(self, batch_size=None):
        raise NotImplementedError
