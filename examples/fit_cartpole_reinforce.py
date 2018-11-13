from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import REINFORCE, Rollout

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(16, activation="tanh", input_shape=input_shape),
                     Dense(num_actions, activation="softmax")])
policy.compile(loss="categorical_crossentropy", optimizer=Adam(1e-2))

agent = REINFORCE(policy, actions=2)

rollout = Rollout(agent, env)

rewards = []
losses = []

for episode in range(1, 301):
    print("\nEpisode", episode)
    rollout_history = rollout.rollout(verbose=1, learning_batch_size=0)
    agent_history = agent.fit(batch_size=32, verbose=1, reset_memory=False)
    rewards.append(rollout_history["reward_sum"])
    losses.append(agent_history["loss"])

fig, (tax, bax) = plt.subplots(2, 1, sharex="all", figsize=(6, 5))
tax.plot(losses)
tax.set_title("Loss")
tax.grid()
bax.plot(rewards)
bax.set_title("Rewards")
bax.grid()
plt.show()
