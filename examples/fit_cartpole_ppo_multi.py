import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster import PPO, MultiRollout, RolloutConfig, Experience
from trickster.utility import visual

envs = [gym.make("CartPole-v1") for _ in range(8)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                    Dense(16, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(5e-4))

agent = PPO(actor,
            critic,
            actions=2,
            memory=Experience(max_length=10000),
            reward_discount_factor_gamma=0.99,
            entropy_penalty_coef=0.005)

rollout = MultiRollout(agent, envs, rollout_configs=RolloutConfig(max_steps=300))

rewards = []
actor_loss = []
actor_utility = []
actor_kld = []
actor_entropy = []
critic_loss = []

for episode in range(1, 2001):
    rollout.reset()

    roll_history = rollout.rollout(verbose=0, learning_batch_size=0)
    agent_history = agent.fit(batch_size=32, verbose=0, reset_memory=True)

    rewards.append(np.mean(roll_history["reward_sum"]))
    actor_loss.append(np.mean(agent_history["actor_loss"]))
    actor_utility.append(np.mean(agent_history["actor_utility"]))
    actor_kld.append(np.mean(agent_history["actor_kld"]))
    actor_entropy.append(np.mean(agent_history["actor_entropy"]))
    critic_loss.append(np.mean(agent_history["critic_loss"]))

    print("\rEpisode {:>4} RWD {:>3.0f} ALOSS {: >7.4f} UTIL {: >7.4f} KLD {: >7.4f} ENTR {: >7.4f} CRIT {:.4f}".format(
        episode,
        np.mean(rewards[-10:]),
        np.mean(agent_history["actor_loss"]),
        np.mean(agent_history["actor_utility"]),
        np.mean(agent_history["actor_kld"]),
        np.mean(agent_history["actor_entropy"]),
        np.mean(agent_history["critic_loss"])), end="")

    if episode % 10 == 0:
        print()

visual.plot_vectors([rewards, actor_loss, actor_utility, actor_kld, actor_entropy, critic_loss],
                    ["Reward", "Actor Loss", "Actor Utility", "Actor KLD", "Actor Entropy", "Critic Loss"],
                    window_size=10)
