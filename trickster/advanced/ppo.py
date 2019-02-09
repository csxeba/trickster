from collections import defaultdict

import numpy as np
from keras import backend as K

from ..abstract import AgentBase
from ..experience import Experience
from ..utility import numeric


class PPO(AgentBase):

    def __init__(self,
                 actor,
                 critic,
                 actions,
                 memory: Experience,
                 reward_discount_factor_gamma=0.99,
                 gae_factor_lambda=0.95,
                 entropy_penalty_coef=0.005,
                 ratio_clip=0.2,
                 state_preprocessor=None):

        super().__init__(actions, memory, reward_discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.lmbda = gae_factor_lambda
        self.possible_actions_onehot = np.eye(len(self.possible_actions))
        self.entropy_penalty_coef = entropy_penalty_coef
        self.ratio_clip = ratio_clip
        self.values = []
        self.probabilities = []
        self._actor_train_fn = self._make_actor_train_function()

    def _make_actor_train_function(self):
        advantages = K.placeholder(shape=(None,))
        action_onehot = K.placeholder(shape=(None, len(self.possible_actions)))
        old_predictions = K.placeholder(shape=(None, len(self.possible_actions)))

        old_probabilities = K.sum(action_onehot * old_predictions, axis=1)
        old_log_prob = K.log(old_probabilities)

        new_pedictions = self.actor.output
        new_probabilities = K.sum(action_onehot * new_pedictions, axis=1)
        new_log_prob = K.log(new_probabilities)

        approximate_kld = K.mean(old_log_prob - new_log_prob)
        ratio = K.exp(new_log_prob - old_log_prob)
        surrogate_1 = ratio * advantages
        surrogate_2 = K.clip(ratio, 1. - self.ratio_clip, 1 + self.ratio_clip) * advantages
        loss = -K.mean(K.minimum(surrogate_1, surrogate_2))

        entropy = -K.mean(new_probabilities * new_log_prob)
        combined_loss = entropy * self.entropy_penalty_coef + loss
        updates = self.actor.optimizer.get_updates(combined_loss, self.actor.weights)
        return K.function(inputs=[self.actor.input, old_predictions, advantages, action_onehot],
                          outputs=[loss, entropy, approximate_kld, combined_loss],
                          updates=updates)

    def sample(self, state, reward):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.possible_actions, p=probabilities, size=1))
        value = self.critic.predict(preprocessed_state)

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
            self.values.append(value)
            self.probabilities.append(probabilities)

        return action

    def push_experience(self, final_state, final_reward, done=True):
        final_value = np.squeeze(self.critic.predict(self.preprocess(final_state)[None, ...]))

        S = np.array(self.states)
        V = np.array(self.values + [final_value])
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [final_reward])
        F = np.zeros(len(S), dtype=bool)
        F[-1] = done
        P = np.array(self.probabilities)

        returns = numeric.compute_gae(R, V[:-1], V[1:], F, self.gamma, self.lmbda)

        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probabilities = []

        self.memory.remember(S, P, A, returns)

    def fit(self, epochs=3, batch_size=32, verbose=1, reset_memory=True):
        history = defaultdict(list)

        for epoch in range(epochs):
            for S, S_, P, A, returns in self.memory.stream(size=batch_size, infinite=False):

                S = self.preprocess(S)
                critic_loss = self.critic.train_on_batch(S, returns)

                action_onehot = self.possible_actions_onehot[A]
                value = np.squeeze(self.critic.predict(S))
                advantage = returns - value
                loss, entropy, approximate_kld, combined_loss = self._actor_train_fn([S, P, advantage, action_onehot])

                history["actor_loss"].append(combined_loss)
                history["actor_utility"].append(loss)
                history["actor_entropy"].append(entropy)
                history["actor_kld"].append(approximate_kld)
                history["critic_loss"].append(critic_loss)

        if reset_memory:
            self.memory.reset()

        return history
