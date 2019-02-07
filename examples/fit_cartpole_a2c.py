import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import A2C, MultiRollout, RolloutConfig, Experience

envs = [gym.make("CartPole-v1") for _ in range(8)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(24, activation="tanh", input_shape=input_shape, kernel_initializer="he_uniform"),
                    Dense(24, activation="tanh", kernel_initializer="he_uniform"),
                    Dense(num_actions, activation="softmax", kernel_initializer="he_uniform")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(24, activation="tanh", input_shape=input_shape, kernel_initializer="he_uniform"),
                     Dense(24, activation="tanh", kernel_initializer="he_uniform"),
                     Dense(1, activation="linear", kernel_initializer="he_uniform")])
critic.compile(loss="mse", optimizer=Adam(5e-4))

agent = A2C(actor, critic, actions=2, memory=Experience(max_length=10000),
            reward_discount_factor=0.98, entropy_penalty_coef=0.)

rollout = MultiRollout(agent, envs, rollout_configs=RolloutConfig(max_steps=200))

rewards = []
actor_losses = []
actor_entropy = []
critic_losses = []

for episode in range(1, 1001):
    rollout.reset()
    episode_rewards = []
    episode_a_losses = []
    episode_a_entropy = []
    episode_c_losses = []
    while not rollout.finished:
        roll_history = rollout.roll(steps=4, verbose=0, learning_batch_size=64)
        episode_rewards.append(roll_history["reward_sum"])
        agent.memory.reset()
        if "actor_utility" in roll_history and "critic_loss" in roll_history:
            episode_a_losses.append(roll_history["actor_utility"])
            episode_c_losses.append(roll_history["critic_loss"])

    rewards.append(sum(episode_rewards))
    actor_losses.append(sum(episode_a_losses) / len(episode_a_losses))
    critic_losses.append(sum(episode_c_losses) / len(episode_c_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} ACTR {:.4f} CRIT {:.4f}".format(
        episode, np.mean(rewards[-10:]), actor_losses[-1], np.mean(critic_losses[-10:])), end="")
    if episode % 10 == 0:
        print()

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, sharex="all", figsize=(6, 5))

ax0.plot(actor_losses)
ax0.set_title("Actor Utility")
ax0.grid()

ax1.plot(actor_entropy)
ax1.set_title("Actor Entropy")
ax1.grid()

ax2.plot(critic_losses)
ax2.set_title("Critic Loss")
ax2.grid()

ax3.plot(rewards)
ax3.set_title("Rewards")
ax3.grid()

plt.show()
