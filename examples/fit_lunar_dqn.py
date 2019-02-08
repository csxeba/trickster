import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import DQN, Rollout, RolloutConfig, Experience
from trickster.utility import visual

env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(24, activation="relu", input_shape=input_shape),
                     Dense(24, activation="relu"),
                     Dense(num_actions, activation="linear")])
policy.compile(loss="mse", optimizer=Adam(1e-4))

agent = DQN(policy,
            actions=num_actions,
            memory=Experience(max_length=10000),
            epsilon=1.,
            reward_discount_factor=0.98,
            use_target_network=True)

rollout = Rollout(agent, env, RolloutConfig(max_steps=200))

rewards = []
losses = []

for warmup in range(1, 33):
    rollout.rollout(verbose=0, learning_batch_size=0)

for episode in range(1, 501):
    rollout.reset()
    episode_rewards = []
    episode_losses = []
    while not rollout.finished:
        roll_history = rollout.roll(steps=2, verbose=0, learning_batch_size=64)
        episode_rewards.append(roll_history["reward_sum"])
        episode_losses.append(roll_history["loss"])
        pass
    rewards.append(sum(episode_rewards))
    losses.append(sum(episode_losses) / len(episode_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} LOSS {:.4f} EPS {:>6.2%}".format(
        episode, np.mean(rewards[-10:]), np.mean(losses[-10:]), agent.epsilon), end="")
    agent.epsilon *= 0.995
    agent.epsilon = max(agent.epsilon, 0.01)
    if episode % 10 == 0:
        print()
        agent.push_weights()

visual.plot_vectors([losses, rewards], ["Loss", "Reward"], window_size=10)
