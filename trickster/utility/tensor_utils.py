import tensorflow as tf


@tf.function
def normalize(tensor):
    tensor_mean = tf.reduce_mean(tensor)
    tensor_std = tf.math.reduce_std(tensor)
    tensor = tensor - tensor_mean
    if tensor_std > 0:
        tensor = tensor / tensor_std
    return tensor
