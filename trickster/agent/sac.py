import numpy as np
import keras

from ..abstract import RLAgentBase
from ..utility import kerasic


class SAC(RLAgentBase):

    """Soft Actor-Critic"""

    history_keys = ["actor_loss", "actor_preds", "Qs", "critic_loss"]

    def __init__(self, actor: keras.Model, value_net: keras.Model, q_net1: keras.Model, q_net2: keras.Model,
                 action_space, memory=None, discount_factor_gamma=0.99, state_preprocessor=None,
                 action_noise_sigma=2., action_noise_sigma_decay=0.9999, min_action_noise_sigma=0.1,
                 action_minima=-np.inf, action_maxima=np.inf, polyak_rate=0.01):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.value_net = value_net
        self.q_net1 = q_net1
        self.q_net2 = q_net2
        self.value_target = kerasic.copy_model(value_net)
        self.action_noise_sigma = action_noise_sigma
        self.action_noise_sigma_decay = action_noise_sigma_decay
        self.min_action_noise_sigma = min_action_noise_sigma
        self.action_minima = action_minima
        self.action_maxima = action_maxima
        self.polyak_rate = polyak_rate
        self._combo_model = None  # type: keras.Model

    def fit_critics(self, batch):
        S, S_, _, R, F = batch
        P = self.actor.predict(S_)
        A = np.argmax(P, axis=-1)
        q1_target = self.q_net1.predict(S_, A)
        q2_target = self.q_net2.predict(S_, A)

