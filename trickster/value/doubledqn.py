import numpy as np
from keras.models import Model, model_from_json

from ..experience import Experience
from . import DQN


class DoubleDQN(DQN):

    def __init__(self, model: Model, actions, memory: Experience, reward_discount_factor=0.99,
                 epsilon=0.99, epsilon_decay=1., epsilon_min=0.1, state_preprocessor=None):
        super().__init__(model, actions, memory,
                         reward_discount_factor,
                         epsilon, epsilon_decay, epsilon_min,
                         state_preprocessor)
        self.double = model_from_json(model.to_json())  # type: Model
        self.push_weights(mix_in_ratio=1)

    def fit(self, batch_size=32, verbose=1):
        S, S_, A, R, F = self.memory.sample(batch_size)
        Y = self.model.predict(S)
        Y[range(len(Y)), A] = self.double.predict(S_).max(axis=1) * self.gamma + R
        Y[F, A[F]] = R[F]
        loss = self.model.train_on_batch(S, Y)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        return {"loss": loss}

    def push_weights(self, mix_in_ratio=1.):
        """
        :param mix_in_ratio: mix_in_ratio * new_weights + (1. - mix_in_ratio) * old_weights
        :return:
        """

        if mix_in_ratio >= 1.:
            self.double.set_weights(self.model.get_weights())
            return

        W = []
        diff = 0.
        mix_in_inverse = 1. - mix_in_ratio
        for old, new in zip(self.double.get_weights(), self.model.get_weights()):
            w = mix_in_inverse*old + mix_in_ratio*new
            diff += (np.linalg.norm(old - w) / old.size)
            W.append(w)
        self.model.set_weights(W)
        self.double.set_weights(W)
        return diff / len(W)
