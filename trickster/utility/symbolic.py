import numpy as np
from keras import backend as K
from keras import metrics

from . import spaces


class TemperedSoftmax:

    def __init__(self, temperature=1.):
        if temperature <= 0.:
            raise ValueError("Parameter: temperature has to be greater than 0.")
        self.temperature = temperature

    def __call__(self, tensor):
        return K.softmax(tensor / self.temperature)


def value_bellman_mean_squared_error(bellman_targets, value_predictions):
    return metrics.mean_squared_error(bellman_targets, value_predictions)


def categorical_entropy(softmaxes):
    return -categorical_log_probability(softmaxes)


def diagonal_normal_entropy(mean, log_std):
    return K.sum(0.5 * K.log(2 * np.pi * np.e * K.exp(log_std)**2.), axis=-1)


def categorical_log_probability(softmaxes):
    n_actions = K.int_shape(softmaxes)[1]
    actions = K.argmax(softmaxes, axis=-1)
    action_mask = K.stop_gradient(K.one_hot(actions, num_classes=n_actions))
    probabilities = K.sum(action_mask * softmaxes, axis=-1)
    return K.log(probabilities)


def diagonal_normal_log_probability(actions, mean, log_std):
    dim = K.shape(mean)[1]
    return K.sum((actions - mean) / K.exp(log_std) ** 2. + 2. * log_std, axis=-1) + dim * K.log(2. * np.pi)
