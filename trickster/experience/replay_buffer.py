from typing import Union, List, Dict, Any

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

    def reset(self):
        self.fields_set = set()

    def read(self):
        self.reset()
        return self.data


class Experience:

    def __init__(self,
                 memory_keys: List[str] = None,
                 max_length: Union[int, None] = 10000):

        self.keys: List[str] = memory_keys
        self.memoirs: Dict[str, np.ndarray] = {key: None for key in self.keys}
        self.width: int = len(self.keys)
        self.max_length: int = max_length
        self.N: int = 0

    def _sanitize(self, data: Union[Dict[str, Any], Transition]) -> Dict[str, np.ndarray]:
        as_dict = {}
        if isinstance(data, Transition):
            data = data.read()
            if not data:
                raise ValueError("Transition not completely set!")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    as_dict[key] = value[None, ...]
                else:
                    as_dict[key] = np.array([value])
        else:
            Ns = set(list(map(len, data.values())))
            if len(Ns) != 1:
                raise ValueError("All data elements must be the same length!")
            as_dict = data
        for key, value in data.items():
            if key not in self.keys:
                raise ValueError(f"Memory element not in available keys: '{key}' not in {self.keys}")
        return as_dict

    def store(self, data: Union[Dict[str, Any], Transition]):
        if self.keys is None:
            raise RuntimeError("Uninitialized replay buffer")
        data = self._sanitize(data)
        for key, element in data.items():
            if self.memoirs[key] is not None:
                element = np.concatenate((self.memoirs[key], element), axis=0)
            self.memoirs[key] = element
            if self.max_length is not None:
                self.memoirs[key] = self.memoirs[key][-self.max_length:]
            self.N = len(self.memoirs[key])

    def pop(self):
        data = {}
        for key, value in self.memoirs.items():
            data[key] = value[-1]
            self.memoirs[key] = value[:-1]
        return data

    def reset(self):
        self.memoirs = {key: None for key in self.keys}

    def get_valid_indices(self):
        return list(range(0, self.N))

    def as_dict(self) -> Dict[str, np.ndarray]:
        return self.memoirs
