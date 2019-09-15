from typing import List

import numpy as np


class TensorReshaper:

    def __init__(self, tensors: List[np.ndarray]):
        self.shapes = [tensor.shape for tensor in tensors]
        self.sizes = [np.prod(shape) for shape in self.shapes]
        self.N = sum(self.sizes)

    @staticmethod
    def unfold(weights: List[np.ndarray]):
        output = []
        for W in weights:
            output.append(W.ravel())
        return np.concatenate(output)

    def fold(self, weights: np.ndarray):
        output = []
        start = 0
        for size, shape in zip(self.sizes, self.shapes):
            end = start + size
            W = weights[start:end].reshape(shape)
            output.append(W)
        return output
