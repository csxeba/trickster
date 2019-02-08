import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import DQN, Rollout, RolloutConfig, Experience

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                  Dense(16, activation="relu"),
                  Dense(num_actions, activation="linear")])
ann.compile(loss="mse", optimizer=Adam(1e-3))

agent = DQN(ann,
            actions=2,
            memory=Experience(max_length=10000),
            epsilon=1.,
            reward_discount_factor=0.98,
            use_target_network=True)

rollout = Rollout(agent, env, config=RolloutConfig(max_steps=300))

rewards = []
losses = []

for warmup in range(1, 33):
    rollout.rollout(verbose=0, learning_batch_size=0)

for episode in range(1, 501):
    rollout.reset()
    episode_rewards = []
    episode_losses = []
    while not rollout.finished:
        roll_history = rollout.roll(steps=4, verbose=0, learning_batch_size=64)
        episode_rewards.append(roll_history["reward_sum"])
        episode_losses.append(roll_history["loss"])

    rewards.append(sum(episode_rewards))
    losses.append(sum(episode_losses) / len(episode_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} LOSS {:.4f} EPS {:>6.2%}".format(
        episode, rewards[-1], losses[-1], agent.epsilon), end="")

    agent.epsilon *= 0.992
    agent.epsilon = max(agent.epsilon, 0.01)
    if episode % 5 == 0:
        agent.push_weights()
        print(" Pushed weights to target net!")

fig, (ax0, ax1) = plt.subplots(2, 1, sharex="all", figsize=(6, 5))

ax0.plot(losses, "r-", alpha=0.5)
ax0.plot(np.convolve(losses, np.ones(10) / 10., "valid"), "b-", alpha=0.8)
ax0.set_title("Critic Loss")
ax0.grid()

ax1.plot(rewards, "r-", alpha=0.5)
ax1.plot(np.convolve(rewards, np.ones(10) / 10., "valid"), "b-", alpha=0.8)
ax1.set_title("Rewards")
ax1.grid()

plt.show()
