import numpy as np
import keras

from ..abstract import RLAgentBase
from ..utility import kerasic


class SAC(RLAgentBase):

    """Soft Actor-Critic"""
    """Under construction"""

    history_keys = ["q1_loss", "q2_loss", "v_loss", "v_dist", "actor_loss", "sigma"]

    def __init__(self,
                 actor: keras.Model,
                 value_net: keras.Model,
                 q_net1: keras.Model,
                 q_net2: keras.Model,
                 action_space,
                 memory=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 action_noise_sigma=2.,
                 action_noise_sigma_decay=0.9999,
                 min_action_noise_sigma=0.1,
                 action_minima=-np.inf,
                 action_maxima=np.inf,
                 entropy_weight_alpha=0.1,
                 polyak_rate=0.01):

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
        self.entropy_alpha = entropy_weight_alpha
        self._combo_model = None

    def _build_model_combination(self):
        input_tensor = keras.Input(self.actor.input_shape[-1:])
        actor_out = self.actor(input_tensor)
        # noisy = keras.layers.Lambda(self._add_noise)(actor_out)
        critic_out = self.q_net1([input_tensor, actor_out])
        self._combo_model = keras.Model(input_tensor, critic_out)
        self.q_net1.trainable = False
        self._combo_model.compile(self.actor.optimizer.from_config(self.actor.optimizer.get_config()),
                                  loss=lambda _, y_pred: -y_pred)
        self.q_net1.trainable = True

    def sample(self, state, reward, done):
        probabilities = self.actor.predict(state[None, ...])[0]
        action = np.random.choice(self.action_space, p=probabilities)
        self._push_step_to_direct_memory_if_learning(state, action, reward, done)
        return action

    def update_critics(self, data):
        S, S_, _, R, F = data

        m = len(S)

        P = self.actor.predict(S_)
        A = np.argmax(P, axis=1)

        entropies = -np.log(P[range(m), A])

        q_target = R + self.gamma * (1 - F) * self.value_target.predict(S_)

        q1 = self.q_net1.predict(S, A)
        q2 = self.q_net2.predict(S, A)

        v_target = np.minimum(q1, q2) - self.entropy_alpha * entropies

        q1_loss = self.q_net1.train_on_batch([S, A], q_target)
        q2_loss = self.q_net2.train_on_batch([S, A], q_target)
        v_loss = self.value_net.train_on_batch(S, v_target)

        vd = self.update_value_target()

        return v_loss.mean(), q1_loss.mean(), q2_loss.mean(), entropies.mean(), vd

    def update_actor(self, data):
        S = data[0]

        self.q_net1.trainable = False
        actor_loss = self._combo_model.train_on_batch(S, S)
        self.q_net1.trainable = True

        return actor_loss.mean()

    def update_value_target(self):
        return kerasic.meld_weights(self.value_target, self.value_net, mix_in_ratio=self.polyak_rate)

    def fit(self, updates=1, batch_size=32):
        actor_losses = []
        q1_losses = []
        q2_losses = []
        v_losses = []
        v_dists = []

        for i in range(1, updates+1):
            v_loss, q1_loss, q2_loss, entropy, vd = self.update_critics(data=self.memory_sampler.sample(batch_size))
            q1_losses.append(q1_loss)
            q2_losses.append(q2_loss)
            v_losses.append(v_loss)
            v_dists.append(vd)

            actor_loss = self.update_actor(data=self.memory_sampler.sample(batch_size))
            actor_losses.append(actor_loss)

        result = {"sigma": self.action_noise_sigma, "actor_loss": np.mean(actor_losses),
                  "q1_loss": np.mean(q1_losses), "q2_loss": np.mean(q2_losses),
                  "v_loss": np.mean(v_losses), "v_dist": np.mean(v_dists)}

        return result

    def get_savables(self) -> dict:
        pass
