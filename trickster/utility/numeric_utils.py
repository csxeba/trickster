import numpy as np


def safe_normalize(vector: np.ndarray):
    vector = vector - vector.mean()
    std = vector.std()
    if std > 0:
        vector /= std
    return vector
