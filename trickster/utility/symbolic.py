from keras import backend as K
from keras import metrics


class TemperedSoftmax:

    def __init__(self, temperature=1.):
        if temperature <= 0.:
            raise ValueError("Parameter: temperature has to be greater than 0.")
        self.temperature = temperature

    def __call__(self, tensor):
        return K.softmax(tensor / self.temperature)


def value_bellman_mean_squared_error(bellman_targets, value_predictions):
    return metrics.mean_squared_error(bellman_targets, value_predictions)


def policy_loss(probabilities, advantages, entropy_penalty_coef=0.):
    log_probabilities = K.log(probabilities)
    entropy = K.sum(log_probabilities)
    policy_gradient = K.mean(log_probabilities * advantages)
    return policy_gradient + entropy * entropy_penalty_coef
