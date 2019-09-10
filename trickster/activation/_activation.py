import keras


class LogSoftMax(keras.layers.Softmax):

    def call(self, inputs):
        softmaxes = super().call(inputs)
        return keras.backend.log(softmaxes)
