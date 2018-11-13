import numpy as np
from keras.models import Model
from keras import backend as K

from ..abstract import AgentBase
from trickster.experience.experience import Experience


class DQN(AgentBase):

    def __init__(self, model: Model, actions, memory: Experience, reward_discount_factor=0.99,
                 epsilon=0.99, epsilon_decay=0.99, epsilon_min=0.1):
        super().__init__(actions, memory, reward_discount_factor)
        self.model = model
        self.output_dim = model.output_shape[1]
        self.action_indices = np.arange(self.output_dim)
        self.Q = []
        self.gamma = reward_discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.train_function = self._make_train_fn()
        self.eye = np.eye(self.output_dim)

    def _make_train_fn(self):
        q_predicted = self.model.output

        action_onehots = K.placeholder([None, self.output_dim], name="action_onehots")
        td_targets = K.placeholder([None], name="td_targets")

        q_values = K.sum(action_onehots * q_predicted, axis=1)
        loss = K.mean(K.square(td_targets - q_values))

        updates = self.model.optimizer.get_updates(loss, self.model.trainable_weights)

        train_fn = K.function(inputs=[self.model.input,
                                      action_onehots,
                                      td_targets],
                              outputs=[loss],
                              updates=updates)
        return train_fn

    def sample(self, state, reward):
        self.states.append(state)
        self.rewards.append(reward)
        Q = self.model.predict(self.preprocess(state)[None, ...])[0]
        action = np.argmax(Q) if np.random.random() > self.epsilon else np.random.randint(0, len(Q))
        self.Q.append(Q)
        self.actions.append(action)
        return action

    def fit(self, batch_size=32, verbose=1):
        S, Y = self.memory.sample(batch_size)
        loss = self.model.train_on_batch(S, Y)
        if verbose:
            print("Loss: {:.4f}".format(loss))
        return loss


if __name__ == '__main__':
    import gym
    from keras.models import Sequential
    from keras.layers import Dense
    from matplotlib import pyplot as plt

    EPISODES = 2000

    env = gym.make("CartPole-v1")
    ann = Sequential([Dense(24, activation="relu", input_shape=env.observation_space.shape,
                            kernel_initializer="he_uniform"),
                      Dense(24, activation="relu", kernel_initializer="he_uniform"),
                      Dense(2, activation="linear", kernel_initializer="he_uniform")])
    ann.compile("adam", "mse")
    dqn = DQN(ann, [0, 1], Experience(max_length=2000), reward_discount_factor=0.99, epsilon=1.,
              epsilon_decay=0.999, epsilon_min=0.01)
    rewards = []
    losses = []
    loss = 0
    reward_sum = 0
    for i in range(1, EPISODES+1):
        print("\rEpisode {:>4} eps: {:>7.2%}, loss: {:.4f}, rwd: {}".format(i, dqn.epsilon, loss, reward_sum),
              end="")
        reward_sum = dqn.rollout(max_steps=140, verbose=0)
        rewards.append(reward_sum)
        if i >= 10:
            loss = dqn.fit(batch_size=32, verbose=0)
            losses.append(loss)
    print()
    fig, (tax, bax) = plt.subplots(2, 1, figsize=(5, 6))
    tax.plot(range(1, len(rewards)+1), rewards)
    bax.plot(range(10, len(losses)+10), losses)
    tax.set_title("rewards")
    bax.set_title("losses")
    tax.grid()
    bax.grid()
    plt.tight_layout()
    plt.show()
