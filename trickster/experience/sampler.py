from typing import List, Union
from .experience import Experience

import numpy as np


class ExperienceSampler:

    def __init__(self, memories: Union[List[Experience], Experience]):
        if isinstance(memories, Experience):
            memories = [memories]
        self.memories = memories

    def reset(self):
        for memory in self.memories:
            memory.reset()

    def sample(self, size=32):
        N = sum(memory.N for memory in self.memories)
        if N < 2:
            return [[]] * self.width
        if size <= 0:
            size = N-1
        valid_indices = self._get_valid_indices()
        num_valid = len(valid_indices)
        size = min(size, num_valid)
        idx = valid_indices[np.random.randint(num_valid, size=size)]
        return self._sample_data(idx)

    def stream(self, size=32, infinite=False):
        arg = self._get_valid_indices()
        while 1:
            np.random.shuffle(arg)
            for start in range(0, len(arg), size):
                idx = arg[start:start+size]
                yield self._sample_data(idx)
            if not infinite:
                break

    def _get_valid_indices(self):
        memories_considered = [i for i, memory in enumerate(self.memories) if memory.N > 2]
        valid_indices = []
        for i in memories_considered:
            valid_indices.extend([[i, j] for j in self.memories[i].get_valid_indices()])
        return np.array(valid_indices)

    def _sample_data(self, indices: np.ndarray):
        sample = {"states": [], "next_states": []}
        sample.update({i: [] for i in range(1, self.width)})

        for i, memory in enumerate(self.memories):
            idx = indices[indices[:, 0] == i][:, 1]
            sample["states"].append(memory.memoirs[0][idx])
            sample["next_states"].append(memory.memoirs[0][idx + 1])
            for j, tensor in enumerate(memory.memoirs[1:], start=1):
                sample[j].append(tensor[idx])

        sample = {key: np.concatenate(value, axis=0) for key, value in sample.items()}
        result = [sample["states"], sample["next_states"]]
        if self.width > 1:
            result += [sample[i] for i in range(1, self.width)]
        return result

    @property
    def width(self):
        return self.memories[0].width
