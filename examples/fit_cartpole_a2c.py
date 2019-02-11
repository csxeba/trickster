import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import A2C, Rollout, RolloutConfig, Experience
from trickster.utility import visual

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                    Dense(16, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(5e-4))

agent = A2C(actor,
            critic,
            action_space=2,
            memory=Experience(max_length=10000),
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.)

rollout = Rollout(agent, env, config=RolloutConfig(max_steps=300))

rewards = []
actor_losses = []
actor_entropy = []
critic_losses = []

for episode in range(1, 2001):
    rollout._reset()
    episode_rewards = []
    episode_a_losses = []
    episode_a_entropy = []
    episode_c_losses = []
    while 1:
        roll_history = rollout.roll(steps=2, verbose=0, learning_batch_size=0)
        if rollout.finished:
            break
        agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)
        episode_rewards.append(roll_history["reward_sum"])
        episode_a_losses.append(agent_history["actor_utility"])
        episode_c_losses.append(agent_history["critic_loss"])
        agent.memory._reset()

    rewards.append(sum(episode_rewards))
    actor_losses.append(sum(episode_a_losses) / len(episode_a_losses))
    critic_losses.append(sum(episode_c_losses) / len(episode_c_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} ACTR {:.4f} CRIT {:.4f}".format(
        episode, np.mean(rewards[-10:]), np.mean(actor_losses[-10:]), np.mean(critic_losses[-10:])), end="")
    print()

visual.plot_vectors([rewards, actor_losses, critic_losses],
                    ["Reward", "Actor Utility", "Critic Loss"],
                    window_size=10)
