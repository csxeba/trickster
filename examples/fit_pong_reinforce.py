"""
Trying to replicate the Pong experiment of Karpathy.
"""

from collections import deque

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import RMSprop

from trickster.rollout import MultiTrajectory, Trajectory
from trickster.agent import REINFORCE
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
        return state  # 80 x 80

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


envs = [FakeEnv() for _ in range(10)]
test_env = FakeEnv()

actor = Sequential([  # 200, 160
    Flatten(input_shape=test_env.shape),
    Dense(200, activation="relu"),
    Dense(2, activation="softmax")
])
actor.compile(RMSprop(1e-4, rho=0.99), "categorical_crossentropy")

agent = REINFORCE(actor, 2, Experience(), discount_factor_gamma=0.99,
                  state_preprocessor=None)

rollout = MultiTrajectory(agent, envs)
test_rollout = Trajectory(agent, test_env)

rewards = deque(maxlen=10)
actor_loss = deque(maxlen=80)
actor_utility = deque(maxlen=80)
actor_entropy = deque(maxlen=80)
critic_loss = deque(maxlen=80)

episode = 0

while 1:

    episode += 1

    history = rollout.rollout(verbose=0, push_experience=True)
    agent_history = agent.fit(batch_size=-1, verbose=0)

    rewards.append(history["mean_reward"])
    actor_loss.append(agent_history["loss"])

    print("\rEpisode {:>4} RWD {:>5.2f} ACTR {:>7.4f}".format(
        episode,
        np.mean(rewards),
        np.mean(actor_loss)), end="")

    if episode % 10 == 0:
        print()
