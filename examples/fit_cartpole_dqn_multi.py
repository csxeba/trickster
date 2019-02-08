import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import DQN, MultiRollout, RolloutConfig, Experience
from trickster.utility import visual

envs = [gym.make("CartPole-v1") for _ in range(4)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

qnet = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                   Dense(16, activation="relu"),
                   Dense(num_actions, activation="linear")])
qnet.compile(loss="mse", optimizer=Adam(1e-3))

agent = DQN(qnet,
            actions=2,
            memory=Experience(max_length=10000),
            epsilon=1.,
            reward_discount_factor=0.98,
            use_target_network=True)

rollout = MultiRollout(agent, envs, rollout_configs=RolloutConfig(max_steps=300))

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
    print("\rEpisode {:>4} RWD {:>3.0f} LOSS {:.4f} EPS {:.2%}".format(
        episode, np.mean(rewards[-10:]), np.mean(losses[-10:]), agent.epsilon), end="")

    agent.epsilon *= 0.992
    agent.epsilon = max(agent.epsilon, 0.01)
    if episode % 5 == 0:
        agent.push_weights()
        print(" Pushed weights to target net!")


visual.plot_vectors([rewards, losses], ["Reward", "Loss"], window_size=10)
