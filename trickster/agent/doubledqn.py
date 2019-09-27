import numpy as np
from keras.models import Model

from ..experience import Experience
from .dqn import DQN


class DoubleDQN(DQN):

    def __init__(self,
                 model: Model,
                 action_space,
                 memory: Experience=None,
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

    def fit(self, updates=1, batch_size=32, polyak_rate=0.1):

        losses = []
        max_q_predictions = []

        for update in range(1, updates+1):

            S, S_, A, R, F = self.memory_sampler.sample(batch_size)

            m = len(S)
            x_index = np.arange(m)

            Qs = self.model.predict(S_)
            next_state_actions = Qs.argmax(axis=1)
            max_q_predictions.append(Qs[x_index, next_state_actions])

            target_Qs = self.target_network.predict(S_)
            bellman_reserve = target_Qs[x_index, next_state_actions]

            bellman_target = self.model.predict(S)
            bellman_target[x_index, A] = bellman_reserve * self.gamma + R
            bellman_target[F, A[F]] = R[F]

            loss = self.model.train_on_batch(S, bellman_target)
            losses.append(loss)

        if polyak_rate:
            self.meld_weights(mix_in_ratio=polyak_rate)

        return {"loss": np.mean(losses), "Qs": np.mean(max_q_predictions), "epsilon": self.epsilon}
