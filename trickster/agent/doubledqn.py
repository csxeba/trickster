import numpy as np
from keras.models import Model

from ..experience import Experience
from .dqn import DQN


class DoubleDQN(DQN):

    history_keys = ["loss", "Qs"]

    def __init__(self,
                 model: Model,
                 action_space,
                 memory: Experience,
                 discount_factor_gamma=0.99,
                 epsilon=0.99,
                 epsilon_decay=1.,
                 epsilon_min=0.1,
                 state_preprocessor=None):

        super().__init__(model=model,
                         action_space=action_space,
                         memory=memory,
                         discount_factor_gamma=discount_factor_gamma,
                         epsilon=epsilon,
                         epsilon_decay=epsilon_decay,
                         epsilon_min=epsilon_min,
                         state_preprocessor=state_preprocessor,
                         use_target_network=True)

    def fit(self, batch_size=32, verbose=1, polyak_tau=0.1):
        S, S_, A, R, F = self.memory_sampler.sample(batch_size)
        target_Qs = self.target_network.predict(S_)
        next_state_actions = self.model.predict(S_).argmax(axis=1)

        x_index = np.arange(len(S))

        bellman_target = self.model.predict(S)
        max_q_values = bellman_target.max(axis=1)
        bellman_reserve = target_Qs[x_index, next_state_actions]
        bellman_target[x_index, A] = bellman_reserve * self.gamma + R
        bellman_target[F, A[F]] = R[F]
        loss = self.model.train_on_batch(S, bellman_target)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        if polyak_tau:
            self.meld_weights(polyak_tau)
        return {"loss": loss, "Qs": max_q_values.mean()}
