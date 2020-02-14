import math

import tensorflow as tf
import tensorflow_probability as tfp


@tf.function(experimental_relax_shapes=True)
def safe_normalize(tensor):
    tensor_mean = tf.reduce_mean(tensor)
    tensor_std = tf.math.reduce_std(tensor)
    tensor = tensor - tensor_mean
    if tensor_std > 0:
        tensor = tensor / tensor_std
    return tensor


def entropy(distribution: tfp.distributions.Distribution):
    if isinstance(distribution, tfp.distributions.MultivariateNormalDiag):
        return 0.5 * tf.math.log(2 * math.pi * math.e * distribution.variance())
    elif isinstance(distribution, tfp.distributions.Categorical):
        return distribution.entropy()
    else:
        raise NotImplementedError
