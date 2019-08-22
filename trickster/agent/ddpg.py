import numpy as np
import keras

from ..abstract import AgentBase
from ..utility import kerasic


class DDPG(AgentBase):

    history_keys = ["actor_loss", "critic_loss"]

    def __init__(self, actor: keras.Model, critic: keras.Model,
                 action_space, memory=None, discount_factor_gamma=0.99, state_preprocessor=None,
                 action_noise_sigma=2., action_noise_sigma_decay=0.9999, min_action_noise_sigma=0.1,
                 action_minima=-np.inf, action_maxima=np.inf):

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
        self._combo_model = None  # type: keras.Model
        self._build_model_combination()

    def _build_model_combination(self):
        input_tensor = keras.Input(self.actor.input_shape[-1:])
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

        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.actions.append(action)

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)  # 0..t
        A = np.array(self.actions)  # 0..t
        R = np.array(self.rewards[1:] + [reward])  # 1..t+1
        F = np.array(self.dones[1:] + [done])

        self._reset_direct_memory()

        self.memory.remember(S, A, R, F)

    def update_critic(self, *, batch_size=None, data=None, update_target_tau=None):
        if data is None and batch_size is None:
            raise ValueError("Please either supply learning data or a batch size for sampling!")
        if batch_size is not None:
            data = self.memory_sampler.sample(batch_size)
        S, S_, A, R, F = data
        A_ = self.actor_target.predict(S_)
        target_Qs = self.critic_target.predict([S_, A_])[..., 0]
        bellman_target = R + (1 - F) * self.gamma * target_Qs
        critic_loss = self.critic.train_on_batch([S, A], bellman_target)
        if update_target_tau:
            self.update_critic_target(update_target_tau)
        return critic_loss

    def update_actor(self, *, batch_size=None, data=None, update_target_tau=None):
        if data is None and batch_size is None:
            raise ValueError("Please either supply learning data or a batch size for sampling!")
        if batch_size is not None:
            data = self.memory_sampler.sample(batch_size)
        S, *_ = data
        self.critic.trainable = False
        actor_loss = self._combo_model.train_on_batch(S, S)
        self.critic.trainable = True
        if update_target_tau:
            self.update_actor_target(update_target_tau)
        return actor_loss

    def fit(self, updates=10, batch_size=32, fit_actor=True, fit_critic=True, update_target_tau=None):
        actor_losses = []
        critic_losses = []
        for i in range(1, updates+1):
            if fit_critic:
                critic_loss = self.update_critic(data=self.memory_sampler.sample(batch_size),
                                                 update_target_tau=update_target_tau)
                critic_losses.append(critic_loss)
            if fit_actor:
                actor_loss = self.update_actor(data=self.memory_sampler.sample(batch_size),
                                               update_target_tau=update_target_tau)
                actor_losses.append(actor_loss)

        result = {}
        if fit_actor:
            result["actor_loss"] = np.mean(actor_losses)
        if fit_critic:
            result["critic_loss"] = np.mean(critic_losses)
        return result

    def update_critic_target(self, tau=0.01):
        kerasic.meld_weights(self.critic_target, self.critic, tau)

    def update_actor_target(self, tau=0.01):
        kerasic.meld_weights(self.actor_target, self.actor, tau)

    def update_targets(self, tau):
        self.update_actor_target(tau)
        self.update_critic_target(tau)
