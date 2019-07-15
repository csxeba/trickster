"""
Trying to replicate the Pong experiment of Karpathy.
"""

from collections import deque

import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from keras.optimizers import Adam

from trickster.rollout import MultiRolling, Trajectory
from trickster.agent import PPO


class FakeEnv:

    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.shape = (80, 80, 2)
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
        state = np.stack([state1, state2], axis=-1)

        return state, reward, done, info

    def reset(self):
        self.initial_state = self.env.reset()
        return self.empty


NUM_PARALLEL_ENVS = 8
NUM_EPISODES = 10000
ROLL_TIMESTEPS = 32
FIT_EPOCHS = 120
FIT_BATCH_SIZE = 32
EXPERIENCE_SIZE = NUM_PARALLEL_ENVS * ROLL_TIMESTEPS
ACTOR_ADAM_LR = 2e-4
CRITIC_ADAM_LR = 1e-4
PPO_RATIO_CLIP = 0.2
DISCOUNT_GAMMA = 0.99
GAE_LAMBDA = 0.97
ENTROPY_PENALTY_BETA = 0.005

envs = [FakeEnv() for _ in range(NUM_PARALLEL_ENVS)]
test_env = FakeEnv()

input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Conv2D(8, 5, strides=2, padding="same", activation="relu", input_shape=[80, 80, 2]),  # 40
                    Conv2D(8, 3, padding="same", activation="relu"),  # 40
                    Conv2D(16, 3, strides=2, padding="same", activation="relu"),  # 20
                    Conv2D(16, 3, padding="same", activation="relu"),  # 20
                    Conv2D(32, 3, strides=2, padding="same", activation="relu"),  # 10
                    GlobalAveragePooling2D(),  # 32
                    Dense(num_actions, activation="softmax")])

actor.compile(loss="categorical_crossentropy", optimizer=Adam(ACTOR_ADAM_LR))

critic = Sequential([Conv2D(8, 5, strides=2, padding="same", activation="relu", input_shape=[80, 80, 2]),  # 40
                    Conv2D(8, 3, padding="same", activation="relu"),  # 40
                    Conv2D(16, 3, strides=2, padding="same", activation="relu"),  # 20
                    Conv2D(16, 3, padding="same", activation="relu"),  # 20
                    Conv2D(32, 3, strides=2, padding="same", activation="relu"),  # 10
                    GlobalAveragePooling2D(),  # 32
                    Dense(1, activation="linear")])

critic.compile(loss="mse", optimizer=Adam(CRITIC_ADAM_LR))

agent = PPO(actor,
            critic,
            action_space=test_env.action_space,
            discount_factor_gamma=DISCOUNT_GAMMA,
            gae_factor_lambda=GAE_LAMBDA,
            entropy_penalty_coef=ENTROPY_PENALTY_BETA)

rollout = MultiRolling(agent.create_workers(NUM_PARALLEL_ENVS), envs)
test_rollout = Trajectory(agent, test_env)

rewards = []
actor_loss = []
actor_utility = []
actor_std = []
actor_kld = []
actor_entropy = []
critic_loss = []

for episode in range(1, NUM_EPISODES+1):
    roll_history = rollout.roll(steps=ROLL_TIMESTEPS, verbose=0, push_experience=True)
    agent_history = agent.fit(epochs=FIT_EPOCHS, batch_size=FIT_BATCH_SIZE, verbose=0, reset_memory=True)

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    rewards.append(test_history["reward_sum"])
    actor_loss.append(np.mean(agent_history["actor_loss"]))
    actor_utility.append(np.mean(agent_history["actor_utility"]))
    actor_std.append(np.mean(agent_history["actor_utility_std"]))
    actor_kld.append(np.mean(agent_history["actor_kld"]))
    actor_entropy.append(np.mean(agent_history["actor_entropy"]))
    critic_loss.append(np.mean(agent_history["critic_loss"]))

    print("\rEpisode {:>4} RWD {:>3.0f} ALOSS {: >7.4f} UTIL {: >7.4f} +- {:>7.4f} kKL {: >7.4f} ENTR {:>7.4f} CRIT {:.4f}".format(
        episode,
        np.mean(rewards[-10:]),
        np.mean(actor_loss[-10:]),
        np.mean(actor_utility[-10:]),
        np.mean(actor_std[-10:]),
        np.mean(actor_kld[-10:]) * 1000,  # kiloKL :D
        np.mean(actor_entropy[-10:]),
        np.mean(critic_loss[-10:])), end="")

    if episode % 10 == 0:
        print()
