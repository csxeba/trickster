import numpy as np
from keras.models import Model
from keras import backend as K

from ..abstract import AgentBase
from ..experience import Experience
from ..utility.numeric import discount_reward
from ..utility.kerasic import copy_model


class A2C(AgentBase):

    def __init__(self,
                 actor: Model,
                 critic: Model,
                 actions,
                 memory: Experience,
                 reward_discount_factor=0.99,
                 state_preprocessor=None,
                 entropy_penalty_coef=0.005,
                 use_target_critic=True,
                 update_critic_every_episode=-1):

        super().__init__(actions, memory, reward_discount_factor, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.action_indices = np.arange(len(self.possible_actions))
        self.possible_actions_onehot = np.eye(len(self.possible_actions))
        self.entropy_penalty_coef = entropy_penalty_coef
        self.target_critic = None
        if use_target_critic:
            assert update_critic_every_episode > 0
            self.target_critic = copy_model(self.critic)

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

    def fit(self, batch_size=-1, verbose=1, reset_memory=True):
        S, S_, A, R, dR, F = self.memory.sample(batch_size)
        assert len(S)

        S_ = self.preprocess(S_)
        value_next = np.squeeze(self.critic.predict(S_))
        bellman_target = value_next * self.gamma + dR
        bellman_target[F] = dR[F]
        mean_bellman_error = self.critic.train_on_batch(S, bellman_target)

        S = self.preprocess(S)
        value = np.squeeze(self.critic.predict(S))
        action_onehot = self.possible_actions_onehot[A]
        advantage = action_onehot * (dR - value)[..., None]
        actor_utility = self.actor.train_on_batch(S, advantage)

        if reset_memory:
            self.memory.reset()
        return {"actor_utility": actor_utility,
                "critic_loss": mean_bellman_error}
