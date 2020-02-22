import math

import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def safe_normalize(tensor):
    tensor_mean = tf.reduce_mean(tensor)
    tensor_std = tf.math.reduce_std(tensor)
    tensor = tensor - tensor_mean
    if tensor_std > 0:
        tensor = tensor / tensor_std
    return tensor


@tf.function(experimental_relax_shapes=True)
def huber_loss(y_true, y_pred):
    l1 = tf.abs(y_true - y_pred)
    l2 = 0.5 * tf.square(y_true - y_pred)
    return tf.minimum(l1, l2)
