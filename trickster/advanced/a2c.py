import numpy as np
from keras.models import Model
from keras import backend as K

from ..abstract import AgentBase
from ..experience import Experience
from ..utility import numeric


class A2C(AgentBase):

    def __init__(self,
                 actor: Model,
                 critic: Model,
                 action_space,
                 memory: Experience,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 entropy_penalty_coef=0.):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.actor = actor
        self.critic = critic
        self.action_indices = np.arange(len(self.possible_actions))
        self.possible_actions_onehot = np.eye(len(self.possible_actions))
        self.entropy_penalty_coef = entropy_penalty_coef
        self._train_function = self._make_actor_train_function()

    def _make_actor_train_function(self):
        advantages = K.placeholder(shape=(None,))
        action_onehot = K.placeholder(shape=(None, len(self.possible_actions)))
        softmaxes = self.actor.output
        probabilities = K.sum(action_onehot * softmaxes, axis=1)
        log_prob = K.log(probabilities)
        entropy = -K.mean(probabilities * log_prob)
        loss = -K.mean(log_prob * advantages)
        combined_utility = entropy * self.entropy_penalty_coef + loss
        updates = self.actor.optimizer.get_updates(combined_utility, self.actor.weights)
        return K.function(inputs=[self.actor.input, advantages, action_onehot],
                          outputs=[loss, entropy, combined_utility],
                          updates=updates)

    def sample(self, state, reward, done):
        preprocessed_state = self.preprocess(state)[None, ...]
        probabilities = self.actor.predict(preprocessed_state)[0]
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities, size=1))

        if self.learning:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [reward])
        F = np.array(self.dones[1:] + [done])

        self._reset_direct_memory()

        self.memory.remember(S, A, R, F)

    def fit(self, batch_size=-1, verbose=1, reset_memory=True):
        S, S_, A, R, F = self.memory.sample(batch_size)
        assert len(S)

        S_ = self.preprocess(S_)
        S = self.preprocess(S)

        value_next = self.critic.predict(S_)[..., 0]
        bellman_target = value_next * self.gamma + R
        bellman_target[F] = R[F]
        mean_bellman_error = self.critic.train_on_batch(S, bellman_target)

        value = self.critic.predict(S)[..., 0]
        action_onehot = self.possible_actions_onehot[A]
        advantage = bellman_target - value

        utility, entropy, loss = self._train_function([S, advantage, action_onehot])

        if reset_memory:
            self.memory.reset()
        return {"actor_utility": utility,
                "actor_entropy": entropy,
                "actor_loss": loss,
                "critic_loss": mean_bellman_error}
