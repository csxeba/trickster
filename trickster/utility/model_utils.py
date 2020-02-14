import tensorflow as tf


@tf.function
def meld_weights(target_model: tf.keras.Model, online_model: tf.keras.Model, mix_in_ratio: float):
    mix_in_inverse = 1. - mix_in_ratio
    for old, new in zip(target_model.weights, online_model.weights):
        old.assign(mix_in_inverse * old + mix_in_ratio * new)
