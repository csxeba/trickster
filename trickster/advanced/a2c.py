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
        self.action_indices = np.arange(len(actions))
        self.possible_actions_onehot = np.eye(len(actions))
        self.values = []

    def sample(self, state, reward):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))
        value = np.squeeze(self.critic.predict(preprocessed_state)[0])

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
            self.values.append(value)
        return action

    def push_experience(self, final_state, final_reward, done=True):
        A = np.array(self.actions)
        self.actions = []

        R = np.array(self.rewards[1:] + [final_reward])
        self.rewards = []

        V = np.array(self.values)
        self.values = []

        S = np.array(self.states)
        self.states = []

        Y = V[1:] * self.gamma + R[:-1]
        if done:
            Y = np.append(Y, final_reward)
        else:
            Y = np.append(Y, R[-1])

        advantages = discount_reward(R, self.gamma) - V
        advantage_weighted_policy_targets = advantages[..., None] * self.possible_actions_onehot[A]

        self.memory.remember(S, Y, advantage_weighted_policy_targets)

    def _fit_critic(self, S, Y, verbose=0):
        loss = self.critic.train_on_batch(S, Y)
        if verbose:
            print("Critic loss: {:.4f}".format(loss))
        return loss

    def _fit_actor(self, S, policy_targets, verbose=0):
        loss = self.actor.train_on_batch(S, policy_targets)
        if verbose:
            print("Actor loss: {:.4f}".format(loss))
        return loss

    def fit(self, batch_size=32, verbose=1, reset_memory=True):
        S, Y, policy_tragets = self.memory.sample(batch_size)
        assert len(S)

        S = self.preprocess(S)

        critic_loss = self._fit_critic(S, Y, verbose)
        actor_loss = self._fit_actor(S, policy_tragets, verbose)

        if reset_memory:
            self.memory.reset()
        return {"actor_loss": actor_loss, "critic_loss": critic_loss}
