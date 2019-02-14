import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.advanced import PPO
from trickster.rollout import Rolling, Rollout, RolloutConfig
from trickster.experience import Experience
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
critic.compile(loss="mse", optimizer=Adam(1e-5))

agent = PPO(actor,
            critic,
            action_space=2,
            memory=Experience(max_length=10000),
            reward_discount_factor_gamma=0.99,
            entropy_penalty_coef=0.005)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Rollout(agent, gym.make("CartPole-v1"))

rewards = []
actor_loss = []
actor_utility = []
actor_kld = []
actor_entropy = []
advantages = []
critic_loss = []

for episode in range(1, 2001):

    rollout.roll(steps=64, verbose=0, push_experience=True)
    agent_history = agent.fit(batch_size=32, verbose=0, reset_memory=True)

    actor_loss.append(np.mean(agent_history["actor_loss"]))
    actor_utility.append(np.mean(agent_history["actor_utility"]))
    actor_kld.append(np.mean(agent_history["actor_kld"]))
    actor_entropy.append(np.mean(agent_history["actor_entropy"]))
    advantages.append(np.mean(agent_history["advantage"]))
    critic_loss.append(np.mean(agent_history["critic_loss"]))

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
    rewards.append(test_history["reward_sum"])

    print("\rEpisode {:>4} RWD {:>5.2f} A {:>5.2f} ACTR {:>7.4f} UTIL {:>7.4f} KKL {:>7.4f} ENTR {:>7.4f} CRIT {:>7.4f}"
          .format(
            episode,
            np.mean(rewards[-10:]),
            np.mean(advantages[-10:]),
            np.mean(actor_loss[-10:]),
            np.mean(actor_utility[-10:]),
            np.mean(actor_kld[-10:])*1000,
            np.mean(actor_entropy[-10:]),
            np.mean(critic_loss[-10:])), end="")
    if episode % 10 == 0:
        print()

visual.plot_vectors([rewards, actor_loss, actor_utility, actor_kld, actor_entropy, critic_loss],
                    ["Reward", "Actor Loss", "Actor Utility", "Actor KLD", "Actor Entropy", "Critic Loss"],
                    window_size=10)
