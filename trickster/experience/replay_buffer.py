from typing import Union, List

import numpy as np


class Transition:

    def __init__(self, keys):
        self.data = {key: None for key in keys}
        self.fields_set = set()

    def set(self, **kwargs):
        for field, value in kwargs.items():
            if value is None:
                continue
            if field in self.fields_set:
                raise RuntimeError("Attempted to write to a field already set!")
            self.data[field] = value
            self.fields_set.add(field)

    @property
    def ready(self):
        return self.fields_set == set(self.data.keys())

    def read(self):
        self.fields_set = set()
        return self.data


class Experience:

    def __init__(self,
                 memory_keys: List[str] = None,
                 max_length: Union[int, None] = 10000):

        self.keys = memory_keys
        self.memoirs = None
        self.width = None
        self.max_length = max_length
        self.final_state = None
        self.N = 0

    def initialize(self, algo):
        if self.keys is None:
            self.keys = algo.memory_keys
        self.memoirs = {key: None for key in self.keys}
        self.width = len(self.keys)

    @staticmethod
    def _sanitize(kwargs):
        Ns = set(list(map(len, kwargs.values())))
        if len(Ns) != 1:
            raise ValueError("All arrays passed to remember() must be the same length!")

    def reset(self):
        self.memoirs = {key: None for key in self.keys}

    def _remember(self, element, key):
        element = np.array(element, copy=False)
        if self.memoirs[key] is not None:
            element = np.concatenate((self.memoirs[key], element), axis=0)
        self.memoirs[key] = element
        if self.max_length is not None:
            self.memoirs[key] = self.memoirs[key][-self.max_length:]
        self.N = len(self.memoirs[key])

    def store_data(self, **kwargs):
        if self.keys is None:
            raise RuntimeError("Uninitialized replay buffer")
        as_trajectories = {key: np.array([element])for key, element in kwargs.items()}
        self._sanitize(as_trajectories)
        for key, element in as_trajectories.items():
            self._remember(element, key)

    def get_valid_indices(self):
        return list(range(0, self.N))
