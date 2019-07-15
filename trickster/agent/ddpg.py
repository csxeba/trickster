import numpy as np
import keras

from ..abstract import AgentBase
from ..utility import kerasic


class DDPG(AgentBase):

    def __init__(self, actor: keras.Model, critic: keras.Model,
                 action_space, memory=None, discount_factor_gamma=0.99, state_preprocessor=None):
        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.actor_target = kerasic.copy_model(actor)
        self.critic_target = kerasic.copy_model(critic)
        self._combo_model = None  # type: keras.Model
        self._build_model_combination()

    def _build_model_combination(self):
        input_tensor = keras.Input(self.actor.input_shape[-1:])
        actor_out = self.actor(input_tensor)
        critic_out = self.critic([input_tensor, actor_out])
        self._combo_model = keras.Model(input_tensor, critic_out)
        self.critic.trainable = False
        self._combo_model.compile(self.actor.optimizer.from_config(self.actor.optimizer.get_config()),
                                  loss=lambda _, y_pred: -keras.backend.mean(y_pred))
        self.critic.trainable = True

    def sample(self, state, reward, done):
        state = self.preprocess(state)
        action = self.actor.predict(state[None, ...])[0]

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

    def fit(self, updates=10, batch_size=32, verbose=1):
        actor_losses = []
        critic_losses = []
        for i, (S, S_, A, R, F) in enumerate(self.memory_sampler.stream(batch_size), start=1):
            target_Qs = self.critic_target.predict([S_, self.actor_target.predict(S_)])[..., 0]
            bellman_target = R + (1 - F) * self.gamma * target_Qs
            target_actions = self.actor_target.predict(S_)
            critic_loss = self.critic.train_on_batch([S, target_actions], bellman_target)
            self.critic.trainable = False
            actor_loss = self._combo_model.train_on_batch(S, S)
            self.critic.trainable = True
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            if i == updates:
                break
        return {"actor_loss": np.mean(actor_losses), "critic_loss": np.mean(critic_losses)}

    def meld_weights(self, actor_ratio=0.15, critic_ratio=0.15):
        kerasic.meld_weights(self.actor_target, self.actor, actor_ratio)
        kerasic.meld_weights(self.critic_target, self.critic, critic_ratio)
