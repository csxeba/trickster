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

    @staticmethod
    def _generate_next_states(idx, memory):
        next_states = []
        for i in idx:
            if i+1 < memory.N:
                next_states.append(memory.memoirs[0][i+1])
            else:
                next_states.append(memory.final_state)
        return np.array(next_states)

    def _sample_data(self, indices: np.ndarray):
        sample = {"states": [], "next_states": []}
        sample.update({i: [] for i in range(1, self.width)})

        for i, memory in enumerate(self.memories):
            if memory.N == 0:
                continue
            idx = indices[indices[:, 0] == i][:, 1]
            if len(idx) == 0:
                continue
            sample["states"].append(memory.memoirs[0][idx])
            sample["next_states"].append(self._generate_next_states(idx, memory))
            for j, tensor in enumerate(memory.memoirs[1:], start=1):
                sample[j].append(tensor[idx])
            pass

        sample = {key: np.concatenate(value, axis=0) for key, value in sample.items()}
        result = [sample["states"], sample["next_states"]]
        if self.width > 1:
            result += [sample[i] for i in range(1, self.width)]
        return result

    @property
    def width(self):
        return self.memories[0].width
