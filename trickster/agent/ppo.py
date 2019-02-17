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
                 action_space,
                 memory: Experience,
                 reward_discount_factor_gamma=0.99,
                 gae_factor_lambda=0.95,
                 entropy_penalty_coef=0.005,
                 ratio_clip_epsilon=0.2,
                 state_preprocessor=None):

        super().__init__(action_space, memory, reward_discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.lmbda = gae_factor_lambda
        self.possible_actions_onehot = np.eye(len(self.possible_actions))
        self.entropy_penalty_coef = entropy_penalty_coef
        self.ratio_clip = ratio_clip_epsilon
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
        utility = -K.mean(K.minimum(surrogate_1, surrogate_2))

        neg_entropy = K.mean(new_log_prob)
        loss = neg_entropy * self.entropy_penalty_coef + utility
        updates = self.actor.optimizer.get_updates(loss, self.actor.weights)
        return K.function(inputs=[self.actor.input, old_predictions, advantages, action_onehot],
                          outputs=[utility, neg_entropy, approximate_kld, loss],
                          updates=updates)

    def sample(self, state, reward, done):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.possible_actions, p=probabilities, size=1))

        if self.learning:
            self.states.append(state)
            self.rewards.append(reward)
            self.actions.append(action)
            self.probabilities.append(probabilities)
            self.dones.append(done)

        return action

    def push_experience(self, state, reward, done):
        final_value = np.squeeze(self.critic.predict(state[None, ...]))
        states = np.array(self.states)
        values = np.squeeze(self.critic.predict(states))
        values = np.append(values, final_value)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards[1:] + [reward])
        dones = np.array(self.dones[1:] + [done])
        probabilities = np.array(self.probabilities)

        returns = numeric.compute_gae(rewards, values[:-1], values[1:], dones, self.gamma, self.lmbda)

        self._reset_direct_memory()
        self.probabilities = []

        self.memory.remember(states, probabilities, actions, returns)

    def fit(self, epochs=3, batch_size=32, verbose=1, reset_memory=True):
        history = defaultdict(list)

        for epoch in range(epochs):
            for state, state_next, probability, action, returns in self.memory.stream(size=batch_size, infinite=False):

                state = self.preprocess(state)
                critic_loss = self.critic.train_on_batch(state, returns)

                action_onehot = self.possible_actions_onehot[action]
                value = np.squeeze(self.critic.predict(state))
                advantage = returns - value

                loss, entropy, approximate_kld, combined_loss = self._actor_train_fn(
                    [state, probability, advantage, action_onehot]
                )

                history["actor_loss"].append(combined_loss)
                history["actor_utility"].append(loss)
                history["actor_entropy"].append(entropy)
                history["actor_kld"].append(approximate_kld)
                history["critic_loss"].append(critic_loss)
                history["advantage"].append(advantage.mean())

        if reset_memory:
            self.memory.reset()

        return history
