import numpy as np
from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import REINFORCE, Rollout, RolloutConfig

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(num_actions, activation="softmax")])
policy.compile(loss="categorical_crossentropy", optimizer=Adam(5e-3))

agent = REINFORCE(policy, action_space=num_actions)

rollout = Rollout(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Rollout(agent, env)

rewards = []
losses = []

for episode in range(1, 501):
    rollout_history = rollout.rollout(verbose=0, push_experience=True)
    agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)
    rewards.append(rollout_history["reward_sum"])
    losses.append(agent_history["loss"])
    print("\rEpisode {:>4} RWD: {:>6.1f}, UTILITY: {: >8.4f}".format(
        episode, np.mean(rewards[-10:]), np.mean(losses[-10:])), end="")
    if episode % 10 == 0:
        print()

fig, (tax, bax) = plt.subplots(2, 1, sharex="all", figsize=(6, 5))
tax.plot(losses, "r-", alpha=0.5)
tax.plot(np.convolve(losses, np.ones(10) / 10., mode="valid"), "b-", alpha=0.8)
tax.set_title("Loss")
tax.grid()

bax.plot(rewards, "r-", alpha=0.5)
bax.plot(np.convolve(rewards, np.ones(10) / 10., mode="valid"), "b-", alpha=0.8)
bax.set_title("Rewards")
bax.grid()

plt.show()

