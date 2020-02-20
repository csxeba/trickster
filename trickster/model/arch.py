import tensorflow as tf


class Architecture(tf.keras.Model):

    """Composes a backbone (ie. hidden layers) and a head (stochastic or deterministic)"""

    def __init__(self,
                 backbone_model: tf.keras.Model,
                 head_model: tf.keras.Model):

        super().__init__()
        self.backbone_model = backbone_model
        self.head_model = head_model
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.num_outputs = head_model.num_outputs

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=None, mask=None):
        x = self.backbone_model(x)
        x = self.head_model(x, training)
        return x
