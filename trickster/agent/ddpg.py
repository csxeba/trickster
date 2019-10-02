import numpy as np
from tensorflow import keras

from ..abstract import RLAgentBase
from ..utility import kerasic


class DDPG(RLAgentBase):

    history_keys = ["actor_loss", "actor_ds", "actor_preds", "Qs", "target_preds", "critic_loss", "critic_ds", "sigma"]

    def __init__(self, actor: keras.Model, critic: keras.Model,
                 action_space, memory=None, discount_factor_gamma=0.99, state_preprocessor=None,
                 action_noise_sigma=2., action_noise_sigma_decay=0.9999, min_action_noise_sigma=0.1,
                 action_minima=-np.inf, action_maxima=np.inf, polyak_rate=0.01):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.actor_target = kerasic.copy_model(actor)
        self.critic_target = kerasic.copy_model(critic)
        self.action_noise_sigma = action_noise_sigma
        self.action_noise_sigma_decay = action_noise_sigma_decay
        self.min_action_noise_sigma = min_action_noise_sigma
        self.action_minima = action_minima
        self.action_maxima = action_maxima
        self.polyak_rate = polyak_rate
        self._combo_model = None  # type: keras.Model
        self._build_model_combination()

    def _build_model_combination(self):
        input_tensor = keras.Input(self.actor.input_shape[1:])
        actor_out = self.actor(input_tensor)
        critic_out = self.critic([input_tensor, actor_out])
        self._combo_model = keras.Model(input_tensor, critic_out)
        self.critic.trainable = False
        self._combo_model.compile(self.actor.optimizer.from_config(self.actor.optimizer.get_config()),
                                  loss=lambda _, y_pred: -y_pred)
        self.critic.trainable = True

    def sample(self, state, reward, done):
        state = self.preprocess(state)
        action = self.actor.predict(state[None, ...])[0]

        if self.learning:
            noise = np.random.normal(loc=0.0, scale=self.action_noise_sigma)
            if self.action_noise_sigma > self.min_action_noise_sigma:
                self.action_noise_sigma *= self.action_noise_sigma_decay
            else:
                self.action_noise_sigma = self.min_action_noise_sigma

            action += noise
            action = np.clip(action, self.action_minima, self.action_maxima)

        self._push_step_to_direct_memory_if_learning(state, action, reward, done)

        return action

    def update_critic(self, *, batch_size=None, data=None):
        if data is None and batch_size is None:
            raise ValueError("Please either supply learning data or a batch size for sampling!")
        if batch_size is not None:
            data = self.memory_sampler.sample(batch_size)
        S, S_, A, R, F = data
        A_ = self.actor_target.predict(S_)
        target_Qs = self.critic_target.predict([S_, A_])[..., 0]
        bellman_target = R + self.gamma * target_Qs
        bellman_target[F] = R[F]
        critic_loss = self.critic.train_on_batch([S, A], bellman_target)
        d = self.update_critic_target()
        return critic_loss, target_Qs.mean(), A_.mean(), d

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
        d = self.update_actor_target()
        return actor_loss, np.linalg.norm(actions, axis=-1).mean(), d

    def fit(self, updates=1, batch_size=32, fit_actor=True, fit_critic=True):
        actor_losses = []
        critic_losses = []
        actor_preds = []
        Qs = []
        target_preds = []
        actor_ds = []
        critic_ds = []

        for i in range(1, updates+1):
            if fit_critic:
                critic_loss, Q, target_pred, critic_d = self.update_critic(data=self.memory_sampler.sample(batch_size))
                critic_losses.append(critic_loss)
                Qs.append(Q)
                target_preds.append(target_pred)
                critic_ds.append(np.sqrt(critic_d))
            if fit_actor:
                actor_loss, actor_pred, actor_d = self.update_actor(data=self.memory_sampler.sample(batch_size))
                actor_losses.append(actor_loss)
                actor_preds.append(actor_pred)
                actor_ds.append(np.sqrt(actor_d))

        result = {"sigma": self.action_noise_sigma}
        if fit_actor:
            result["actor_loss"] = np.mean(actor_losses)
            result["actor_preds"] = np.mean(actor_preds)
            result["actor_ds"] = np.mean(actor_ds)
        if fit_critic:
            result["critic_loss"] = np.mean(critic_losses)
            result["Qs"] = np.mean(Qs)
            result["target_preds"] = np.mean(target_preds)
            result["critic_ds"] = np.mean(critic_ds)

        return result

    def update_critic_target(self):
        return kerasic.meld_weights(self.critic_target, self.critic, self.polyak_rate)

    def update_actor_target(self):
        return kerasic.meld_weights(self.actor_target, self.actor, self.polyak_rate)

    def update_targets(self):
        actor_d = self.update_actor_target()
        critic_d = self.update_critic_target()
        return actor_d, critic_d

    def get_savables(self) -> dict:
        return {"DDPG_actor": self.actor, "DDPG_critic": self.critic}
