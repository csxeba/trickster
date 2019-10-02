import numpy as np
from tensorflow import keras

from ..abstract import RLAgentBase
from ..utility from tensorflow import kerasic, symbolic


class SAC(RLAgentBase):

    """Soft Actor-Critic"""
    """Under construction"""

    history_keys = ["actor_loss", "actor_preds", "Qs", "critic_loss"]

    def __init__(self,
                 actor: keras.Model,
                 value_net: keras.Model,
                 q_net1: keras.Model,
                 q_net2: keras.Model,
                 action_space,
                 memory=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 entropy_weight_alpha=0.1,
                 polyak_rate=0.01):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.value_net = value_net
        self.q_net1 = q_net1
        self.q_net2 = q_net2
        self.value_target = kerasic.copy_model(value_net)
        self.polyak_rate = polyak_rate
        self.entropy_alpha = entropy_weight_alpha
        self._combo_model = None  # type: keras.Model

    def sample(self, state, reward, done):
        state = self.preprocess(state)
        probabilities = self.actor.predict(state[None, ...])[0]
        if self.learning:
            action = np.random.choice(self.action_space, p=probabilities)
        else:
            action = np.argmax(probabilities)

        self._push_step_to_direct_memory_if_learning(state, action, reward, done)

    def _actor_entropy(self):
        return ...

    def fit_critics(self, batch):
        S, S_, _, R, F = batch

        A = self.actor.predict(S_)

        q1_target = self.q_net1.predict(S_, A)
        q2_target = self.q_net2.predict(S_, A)

        q_target = np.minimum(q1_target, q2_target) + self.entropy_alpha * self._actor_entropy()
