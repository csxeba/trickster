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

    @property
    def N(self):
        return sum(memory.N for memory in self.memories)

    def sample(self, size=32):
        if self.N == 0:
            return [[]] * (self.width+1)
        if size <= 0:
            size = self.N
        valid_indices = self._get_valid_indices()
        num_valid = len(valid_indices)
        size = min(size, num_valid)
        if size < num_valid:
            idx = valid_indices[np.random.randint(0, num_valid, size=size)]
        else:
            idx = valid_indices
        sample = self._sample_data(idx)
        return sample

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
        memories_considered = [i for i, memory in enumerate(self.memories) if memory.N >= 1]
        valid_indices = []
        for i in memories_considered:
            valid_indices.extend([[i, j] for j in self.memories[i].get_valid_indices()])
        return np.array(valid_indices)

    def _sample_data(self, indices: np.ndarray):
        sample = [[] for _ in range(self.width)]

        for i, memory in enumerate(self.memories):
            if memory.N == 0:
                continue
            idx = indices[indices[:, 0] == i][:, 1]
            if len(idx) == 0:
                continue
            for j, tensor in enumerate(memory.memoirs):
                sample[j].append(tensor[idx])

        sample = [np.concatenate(s, axis=0) for s in sample]
        return sample

    @property
    def width(self):
        w = None
        for m in self.memories:
            w = m.width
            if w > 0:
                break
        return w
