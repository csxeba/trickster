import numpy as np
from keras.models import Model

from ..abstract import AgentBase
from ..experience import Experience
from ..utility.numeric import discount_reward


class A2C(AgentBase):

    def __init__(self,
                 actor: Model,
                 critic: Model,
                 actions,
                 memory: Experience,
                 reward_discount_factor=0.99,
                 state_preprocessor=None):

        super().__init__(actions, memory, reward_discount_factor, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.action_indices = np.arange(len(self.possible_actions))
        self.possible_actions_onehot = np.eye(len(self.possible_actions))

    def sample(self, state, reward):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
        return action

    def push_experience(self, final_state, final_reward, done=True):
        S = np.array(self.states)
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [final_reward])
        dR = discount_reward(R, self.gamma)
        F = np.zeros(len(S), dtype=bool)
        F[-1] = done

        self.states = []
        self.actions = []
        self.rewards = []

        self.memory.remember(S, A, R, dR, F)

    def _fit_critic(self, S, critic_targets, verbose=0):
        loss = self.critic.train_on_batch(S, critic_targets)
        if verbose:
            print("Critic loss: {:.4f}".format(loss))
        return loss

    def _fit_actor(self, S, policy_targets, verbose=0):
        loss = self.actor.train_on_batch(S, policy_targets)
        if verbose:
            print("Actor loss: {:.4f}".format(loss))
        return loss

    def fit(self, batch_size=32, verbose=1, reset_memory=True):
        S, S_, A, R, dR, F = self.memory.sample(batch_size)
        assert len(S)

        S = self.preprocess(S)
        S_ = self.preprocess(S_)

        critic_targets = np.squeeze(self.critic.predict(S_)) * self.gamma + R
        critic_targets[F] = R[F]

        values = np.squeeze(self.critic.predict(S))
        advantages = dR - values
        policy_targets = self.possible_actions_onehot[A] * advantages[..., None]

        critic_loss = self._fit_critic(S, critic_targets, verbose)
        actor_loss = self._fit_actor(S, policy_targets, verbose)

        if reset_memory:
            self.memory.reset()
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
