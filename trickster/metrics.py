from tensorflow import keras


def entropy(action_onehots, action_log_probabilities):
    return keras.backend.sum(action_onehots * action_log_probabilities, axis=-1)


def kl_divergence(old_action_log_probabilities, new_action_log_probabilities):
    return keras.backend.sum(old_action_log_probabilities - new_action_log_probabilities, axis=-1)
