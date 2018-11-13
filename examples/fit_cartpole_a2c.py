from matplotlib import pyplot as plt

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from trickster import A2C, MultiRollout, RolloutConfig, Experience

envs = [gym.make("CartPole-v1") for _ in range(8)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(24, activation="relu", input_shape=input_shape, kernel_initializer="he_uniform"),
                    Dense(num_actions, activation="softmax", kernel_initializer="he_uniform")])
actor.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-5, momentum=0.9))

critic = Sequential([Dense(24, activation="relu", input_shape=input_shape, kernel_initializer="he_uniform"),
                     Dense(24, activation="relu", kernel_initializer="he_uniform"),
                     Dense(1, activation="linear", kernel_initializer="he_uniform")])
critic.compile(loss="mse", optimizer=SGD(lr=1e-5, momentum=0.9))

agent = A2C(actor, critic, actions=2, memory=Experience(max_length=10000), reward_discount_factor=0.99)

rollout = MultiRollout(agent, envs, rollout_configs=RolloutConfig(max_steps=300))

rewards = []
actor_losses = []
critic_losses = []

for episode in range(1, 1001):
    rollout.reset()
    episode_rewards = []
    episode_a_losses = []
    episode_c_losses = []
    while not rollout.finished:
        roll_history = rollout.roll(steps=4, verbose=0, learning_batch_size=32)
        episode_rewards.append(roll_history["reward_sum"])
        if "actor_loss" in roll_history and "critic_loss" in roll_history:
            episode_a_losses.append(roll_history["actor_loss"])
            episode_c_losses.append(roll_history["critic_loss"])

    rewards.append(sum(episode_rewards))
    actor_losses.append(sum(episode_a_losses) / len(episode_a_losses))
    critic_losses.append(sum(episode_c_losses) / len(episode_c_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} ACTR {:.4f} CRIT {:.4f}".format(
        episode, rewards[-1], actor_losses[-1], critic_losses[-1]), end="")

fig, (tax, mx, bax) = plt.subplots(3, 1, sharex="all", figsize=(6, 5))
tax.plot(actor_losses)
tax.set_title("Actor Loss")
tax.grid()
mx.plot(critic_losses)
mx.set_title("Critic Loss")
mx.grid()
bax.plot(rewards)
bax.set_title("Rewards")
bax.grid()
plt.show()
