import numpy as np

from keras.models import Model
from keras.utils import to_categorical

from ..abstract import RLAgentBase
from ..experience import Experience
from ..utility.numeric import discount_reward
from ..utility import symbolic


class REINFORCE(RLAgentBase):

    history_keys = ["loss", "entropy", "kld"]

    def __init__(self,
                 model: Model,
                 action_space,
                 memory: Experience=None,
                 discount_factor_gamma=0.99,
                 state_preprocessor=None,
                 entropy_penalty_coef=0.):

        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)

    def sample(self, state, reward, done):
        probabilities = np.squeeze(self.model.predict(self.preprocess(state)[None, ...]))
        if self.learning:
            action = np.squeeze(np.random.choice(self.action_indices, p=probabilities))
        else:
            action = np.squeeze(np.argmax(probabilities))

        self._push_direct_experience(state, action, reward, done)

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)
        A = np.array(self.actions)
        R = np.array(self.rewards[1:] + [reward])
        R = discount_reward(R, self.gamma)
        F = np.array(self.dones[1:] + [done])

        rstd = R.std()
        if rstd > 0:
            R = (R - R.mean()) / rstd

        self._reset_direct_memory()

        self.memory.remember(S, A, R, dones=F, final_state=state)

    def fit(self, batch_size=-1, reset_memory=True):
        S, _, A, R, F = self.memory_sampler.sample(batch_size)
        m = len(S)

        P = self.model.predict(S)
        Y = to_categorical(A, num_classes=self.output_dim)
        loss = self.model.train_on_batch(S, Y, sample_weight=R)  # works because of the definition of categorical XEnt
        new_probabilities = self.model.predict(S)
        new_log_P = np.log(new_probabilities + 1e-7)

        entropy = -new_log_P.max(axis=1).mean()
        approximate_kld = (np.log(P + 1e-7) - new_log_P).sum(axis=1).mean()
        if reset_memory:
            self.memory.reset()
        return {"loss": loss, "entropy": entropy, "kld": approximate_kld, "batch_size": m}

    def get_savables(self):
        return {"REINFORCE_policy": self.model}
