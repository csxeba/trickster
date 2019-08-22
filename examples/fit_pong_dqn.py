"""
Trying to replicate the Pong experiment of Karpathy.
"""

from collections import deque

import numpy as np
import gym
import keras

from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.agent import DQN
from trickster.experience import Experience


class FakeEnv:

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.shape = (80, 80, 1)
        self.initial_state = None

    @property
    def empty(self):
        return np.zeros(self.shape)

    def _preporcess(self, state):
        state = state[35:195]  # crop
        state = state[::2, ::2, 0]  # downsample by factor of 2
        state[state == 144] = 0  # erase background (background type 1)
        state[state == 109] = 0  # erase background (background type 2)
        state[state != 0] = 1  # everything else (paddles, ball) just set to 1
        return state[..., None]  # 80 x 80

    def step(self, action):
        state1, reward1, done, info = self.env.step(action)
        if done:
            return self.empty, reward1, done, info

        if self.initial_state is not None:
            state2 = self.initial_state
            reward = reward1
            self.initial_state = None
        else:
            state2, reward2, done, info = self.env.step(action)
            reward = max(reward1, reward2)
        if done:
            return self.empty, reward, done, info

        state1, state2 = map(self._preporcess, [state1, state2])
        state = np.abs(state1 - state2)

        return state, reward, done, info

    def reset(self):
        self.initial_state = self.env.reset()
        return self.empty


env = FakeEnv()
test_env = FakeEnv()

qnet = keras.models.Sequential([
    keras.layers.Conv2D(8, 3, padding="same", kernel_initializer="he_uniform", input_shape=test_env.shape),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPool2D(),  # 40
    keras.layers.Conv2D(16, 3, kernel_initializer="he_uniform", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPool2D(),  # 20
    keras.layers.Conv2D(16, 3, kernel_initializer="he_uniform", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPool2D(),  # 10
    keras.layers.Conv2D(16, 3, kernel_initializer="he_uniform", padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.MaxPool2D(),  # 5
    keras.layers.GlobalAveragePooling2D(),  # 16
    keras.layers.Dense(4, kernel_initializer="he_uniform"),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Dense(2, kernel_initializer="he_uniform")
])
qnet.compile(keras.optimizers.Adam(1e-3), "mse")

agent = DQN(qnet, 2, Experience(max_length=10_000), discount_factor_gamma=0.99,
            epsilon=1.0, epsilon_decay=0.99999, epsilon_min=0.3, use_target_network=True,
            state_preprocessor=None)

rollout = Rolling(agent, env, RolloutConfig(skipframes=2))
test_rollout = Trajectory(agent, test_env)

rewards = deque(maxlen=100)
losses = deque(maxlen=100)

episode = 0

while 1:

    episode += 1

    rollout.roll(steps=4, verbose=0, push_experience=True)
    if agent.memory.N < 1000:
        print(f"\rFilling memory... {agent.memory.N}/1000", end="")
        continue

    agent_history = agent.fit(batch_size=32, verbose=0)
    history = test_rollout.rollout(verbose=0, push_experience=False)

    rewards.append(history["reward_sum"])
    losses.append(agent_history["loss"])

    print("\rEpisode {:>4} RWD {:>5.2f} BMLOSS {:>7.4f}, EPS {:.2%}".format(
        episode,
        np.mean(rewards),
        np.mean(losses),
        agent.epsilon), end="")

    agent.meld_weights(mix_in_ratio=0.01)

    if episode % 100 == 0:
        print(" Pushing weights!")
        agent.push_weights()
