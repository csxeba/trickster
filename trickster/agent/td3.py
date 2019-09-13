from typing import List

import numpy as np
import keras

from .ddpg import DDPG
from ..utility import kerasic


def add_noise(inputs):
    x, sigma, clip_noise, clip_action = inputs
    K = keras.backend
    noise = K.clip(K.random_normal(shape=K.int_shape(x), stddev=sigma), -clip_noise, clip_noise)
    return K.clip(x + noise, -clip_action, clip_action)


class TD3(DDPG):

    """Twin Delayed Deep Deterministic Policy Gradient"""

    history_keys = ["actor_loss", "actor_preds", "Qs", "critic1_loss", "critic2_loss"]

    def __init__(self,
                 actor: keras.Model,
                 critics: List[keras.Model],
                 action_space,
                 memory=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 actor_update_delay=2,
                 action_noise_sigma=0.1,
                 action_noise_sigma_decay=1.,
                 min_action_noise_sigma=0.1,
                 target_noise_sigma=0.2,
                 target_noise_clip=0.5,
                 action_minima=-np.inf,
                 action_maxima=np.inf,
                 polyak_rate=0.01):

        self.critic_dupe = critics[1]
        self.critic_dupe_target = kerasic.copy_model(self.critic_dupe)
        self.target_noise_sigma = target_noise_sigma
        self.target_noise_clip = target_noise_clip

        super().__init__(actor, critics[0], action_space, memory, discount_factor_gamma, state_preprocessor,
                         action_noise_sigma, action_noise_sigma_decay, min_action_noise_sigma, action_minima,
                         action_maxima, polyak_rate)
        self.actor_update_delay = actor_update_delay
        self.update_counter = 1

    def _build_model_combination(self):
        input_tensor = keras.Input(self.actor.input_shape[-1:])
        actor_out = self.actor(input_tensor)
        noisy = keras.layers.Lambda(add_noise)(
            [actor_out, self.target_noise_sigma, self.target_noise_clip, self.action_maxima]
        )
        critic_out = self.critic([input_tensor, noisy])
        self._combo_model = keras.Model(input_tensor, critic_out)
        self.critic.trainable = False
        self._combo_model.compile(self.actor.optimizer.from_config(self.actor.optimizer.get_config()),
                                  loss=lambda _, y_pred: -y_pred)
        self.critic.trainable = True

    def update_actor(self, *, batch_size=None, data=None):
        if data is None and batch_size is None:
            raise ValueError("Please either supply learning data or a batch size for sampling!")
        if batch_size is not None:
            data = self.memory_sampler.sample(batch_size)
        S, *_ = data
        self.critic.trainable = False
        actor_loss = self._combo_model.train_on_batch(S, S)
        actions = self.actor.predict(S)
        self.critic.trainable = True
        if self.polyak_rate:
            self.update_actor_target()
        return actor_loss, actions.mean()

    def update_critic(self, *, batch_size=None, data=None):
        if data is None and batch_size is None:
            raise ValueError("Please either supply learning data or a batch size for sampling!")
        if batch_size is not None:
            data = self.memory_sampler.sample(batch_size)
        S, S_, A, R, F = data
        A_ = self.actor_target.predict(S_)
        target_Qs = self.critic_target.predict([S_, A_])[..., 0]
        dupe_target_Qs = self.critic_dupe_target.predict([S_, A_])[..., 0]
        assert not np.equal(target_Qs, dupe_target_Qs)
        bellman_target = R + self.gamma * np.minimum(target_Qs, dupe_target_Qs)
        bellman_target[F] = R[F]
        critic_loss = self.critic.train_on_batch([S, A], bellman_target)
        critic_dupe_loss = self.critic_dupe.train_on_batch([S, A], bellman_target)
        if self.polyak_rate:
            self.update_critic_target()
        return critic_loss, critic_dupe_loss, target_Qs.mean()

    def fit(self, updates=2, batch_size=32, fit_actor=True, fit_critic=True, update_target_tau=0.01):
        if updates % self.actor_update_delay != 0:
            raise ValueError("updates_per_episode must be divisible by actor_update_delay (={}) in TD3"
                             .format(self.actor_update_delay))

        actor_losses = []
        critic_losses = []
        critic_dupe_losses = []
        actor_preds = []
        Qs = []

        for i in range(1, updates+1):
            fit_actor = self.update_counter % self.actor_update_delay == 0
            if fit_critic:
                critic_loss, critic_dupe_loss, Q = self.update_critic(data=self.memory_sampler.sample(batch_size))
                critic_losses.append(critic_loss)
                critic_dupe_losses.append(critic_loss)
                Qs.append(Q)
            if fit_actor:
                actor_loss, actor_pred = self.update_actor(data=self.memory_sampler.sample(batch_size))
                actor_losses.append(actor_loss)
                actor_preds.append(actor_pred)
            self.update_counter += 1

        result = {}
        if fit_actor:
            result["actor_loss"] = np.mean(actor_losses)
            result["actor_preds"] = np.mean(actor_preds)
        if fit_critic:
            result["critic1_loss"] = np.mean(critic_losses)
            result["critic2_loss"] = np.mean(critic_dupe_losses)
            result["Qs"] = np.mean(Qs)

        return result

    def update_critic_target(self):
        super().update_actor_target()
        kerasic.meld_weights(self.critic_dupe_target, self.critic_dupe, self.polyak_rate)

    def get_savables(self) -> dict:
        return {"TD3_actor": self.actor, "TD3_critic": self.critic, "TD3_critic2": self.critic_dupe}
