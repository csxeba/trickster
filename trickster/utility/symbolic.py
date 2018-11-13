from keras import backend as K


class SoftMax2:

    def __init__(self, temperature=1.):
        if temperature <= 0.:
            raise ValueError("Parameter: temperature has to be greater than 0.")
        self.temperature = temperature

    def __call__(self, tensor):
        return K.softmax(tensor / self.temperature)
