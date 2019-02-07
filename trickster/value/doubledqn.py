import numpy as np
from keras.models import Model, model_from_json

from ..experience import Experience
from . import DQN


class DoubleDQN(DQN):

    def __init__(self,
                 model: Model,
                 actions,
                 memory: Experience,
                 reward_discount_factor=0.99,
                 epsilon=0.99,
                 epsilon_decay=1.,
                 epsilon_min=0.1,
                 state_preprocessor=None):

        super().__init__(model=model,
                         actions=actions,
                         memory=memory,
                         reward_discount_factor=reward_discount_factor,
                         epsilon=epsilon,
                         epsilon_decay=epsilon_decay,
                         epsilon_min=epsilon_min,
                         state_preprocessor=state_preprocessor,
                         use_target_network=True)

    def fit(self, batch_size=32, verbose=1, update_target=False):
        S, S_, A, R, F = self.memory.sample(batch_size)
        bellman_target = self.target_network.predict(S_)
        next_state_actions = self.model.predict(S_).argmax(axis=1)

        Q = self.model.predict(S)
        x_index = np.arange(len(Q))
        Q[x_index, A] = bellman_target[x_index, next_state_actions] * self.gamma + R
        Q[F, A[F]] = R[F]
        loss = self.model.train_on_batch(S, Q)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        return {"loss": loss}
