from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import DQN, Rollout, RolloutConfig, Experience

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(num_actions, activation="linear")])
policy.compile(loss="mse", optimizer=Adam(1e-3))

agent = DQN(policy, actions=2, memory=Experience(max_length=10000), epsilon=1., reward_discount_factor=0.98)

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
        pass
    rewards.append(sum(episode_rewards))
    losses.append(sum(episode_losses) / len(episode_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} LOSS {:.4f} EPS {:>6.2%}".format(
        episode, rewards[-1], losses[-1], agent.epsilon))
    agent.epsilon *= 0.995
    agent.epsilon = max(agent.epsilon, 0.01)

fig, (tax, bax) = plt.subplots(2, 1, sharex="all", figsize=(6, 5))
tax.plot(losses)
tax.set_title("Loss")
tax.grid()
bax.plot(rewards)
bax.set_title("Rewards")
bax.grid()
plt.show()
