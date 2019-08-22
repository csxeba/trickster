import numpy as np

from keras.models import Model

from ..abstract import AgentBase
from ..experience import Experience
from ..utility.numeric import discount_reward


class REINFORCE(AgentBase):

    history_keys = ["loss", "entropy", "kld"]

    def __init__(self, model: Model, action_space, memory: Experience=None, discount_factor_gamma=0.99,
                 state_preprocessor=None):
        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.possible_actions_onehot = np.eye(self.output_dim)
        self.probabilities = []

    def sample(self, state, reward, done):
        probabilities = np.squeeze(self.model.predict(self.preprocess(state)[None, ...]))
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities))

        if self.learning:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.probabilities.append(probabilities[action])

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)
        Y = self.possible_actions_onehot[self.actions]
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [reward])
        R = discount_reward(R, self.gamma)
        P = np.array(self.probabilities)

        rstd = R.std()
        if rstd > 0:
            R = (R - R.mean()) / rstd

        self._reset_direct_memory()
        self.probabilities = []

        self.memory.remember(S, A, Y, R[..., None], P)

    def fit(self, batch_size=-1, verbose=1, reset_memory=True):
        S, _, A, Y, R, P = self.memory_sampler.sample(batch_size)
        m = len(S)

        loss = self.model.train_on_batch(S, Y*R)  # works because of the definition of categorical XEnt
        new_probabilities = self.model.predict(S)
        new_log_P = np.log(new_probabilities[range(m), tuple(A)])

        entropy = -np.mean(np.log(np.max(new_probabilities, axis=1)))
        approximate_kld = np.log(P) - new_log_P
        if reset_memory:
            self.memory.reset()
        return {"loss": loss, "entropy": entropy, "kld": approximate_kld, "batch_size": m}
