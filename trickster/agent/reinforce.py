import numpy as np

from keras.models import Model

from trickster.abstract import AgentBase
from trickster.experience import Experience
from trickster.utility.numeric import discount_reward


class REINFORCE(AgentBase):

    def __init__(self, model: Model, action_space, memory: Experience=None, discount_factor_gamma=0.99,
                 state_preprocessor=None):
        super().__init__(action_space, memory, discount_factor_gamma, state_preprocessor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.possible_actions_onehot = np.eye(self.output_dim)

    def sample(self, state, reward, done):
        probabilities = np.squeeze(self.model.predict(self.preprocess(state)[None, ...]))
        action = np.squeeze(np.random.choice(self.action_indices, p=probabilities))

        if self.learning:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

        return action

    def push_experience(self, state, reward, done):
        S = np.array(self.states)
        A = self.possible_actions_onehot[self.actions]
        R = np.array(self.rewards[1:] + [reward])
        R = discount_reward(R, self.gamma)

        rstd = R.std()
        if rstd > 0:
            R = (R - R.mean()) / rstd

        self._reset_direct_memory()

        self.memory.remember(S, A*R[..., None])

    def fit(self, batch_size=-1, verbose=1, reset_memory=True):
        S, _, Y = self.memory.sample(-1)
        loss = self.model.train_on_batch(S, Y)  # works because of the definition of categorical XEnt
        if verbose:
            print("Loss: {:.4f}".format(loss))
        if reset_memory:
            self.memory.reset()
        return {"loss": loss}
