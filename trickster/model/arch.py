import tensorflow as tf


class Architecture(tf.keras.Model):

    """Composes a backbone (ie. hidden layers) and a head (stochastic or deterministic)"""

    def __init__(self,
                 model: tf.keras.Model,
                 optimizer: tf.keras.optimizers.Optimizer = "default"):

        super().__init__()
        self.model = model
        if optimizer == "default":
            optimizer = tf.keras.optimizers.Adam(1e-3)
        self.optimizer = optimizer

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        x = self.model(x, training, mask)
        return x


class TestingModel(tf.keras.Model):

    def __init__(self, output: tf.Tensor):
        super().__init__()
        self.output_tensor = output

    def call(self, inputs, training=None, mask=None):
        return self.output_tensor
