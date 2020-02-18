from typing import Union

import numpy as np
import tensorflow as tf


class EpsilonGreedy:

    def __init__(self,
                 epsilon: float = 1.,
                 epsilon_decay: float = 1.,
                 epsilon_min: float = 0.1):

        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def sample(self, Q, do_update):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(Q))
        else:
            action = np.argmax(Q)
        if do_update:
            self.update()
        return action

    def update(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def reset(self):
        self.epsilon = self.initial_epsilon


class NumericContinuousActionSmoother:

    def __init__(self,
                 sigma: float = 2.,
                 sigma_decay: float = 0.999,
                 sigma_min: float = 0.1,
                 action_minima: Union[float, np.ndarray] = None,
                 action_maxima: Union[float, np.ndarray] = None,):

        self.initial_sigma = sigma
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.action_minima = action_minima
        self.action_maxima = action_maxima

    def sample(self, action, do_update):
        smoothed_action = np.random.normal(loc=action, scale=self.sigma)
        if do_update:
            self.update()
        if self.action_minima is not None:
            smoothed_action = np.maximum(smoothed_action, self.action_minima)
        if self.action_maxima is not None:
            smoothed_action = np.minimum(smoothed_action, self.action_maxima)
        return smoothed_action

    def update(self):
        self.sigma *= self.sigma_decay
        self.sigma = max(self.sigma, self.sigma_min)

    def reset(self):
        self.sigma = self.initial_sigma


@tf.function(experimental_relax_shapes=True)
def add_clipped_noise(action, sigma, noise_clip, action_minima, action_maxima):
    noise = tf.random.normal(shape=action.shape, mean=0.0, stddev=sigma)
    noise = tf.clip_by_value(noise, -noise_clip, noise_clip)
    smoothed_action = action + noise
    smoothed_action = tf.maximum(smoothed_action, action_minima)
    smoothed_action = tf.minimum(smoothed_action, action_maxima)
    return smoothed_action
