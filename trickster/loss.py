from tensorflow import keras


def policy_gradient(action_advantages, action_log_probabilities):
    return keras.losses.categorical_crossentropy(action_advantages, action_log_probabilities)


def mean_bellman_error(target_qs, predicted_qs):
    return keras.losses.mean_squared_error(target_qs, predicted_qs)
