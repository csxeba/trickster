import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.advanced import PPO
from trickster.rollout import MultiRolling, Rollout, RolloutConfig
from trickster.experience import Experience
from trickster.utility import visual

envs = [gym.make("CartPole-v1") for _ in range(16)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                    Dense(16, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(1e-3))

agent = PPO(actor,
            critic,
            action_space=2,
            memory=Experience(max_length=10000),
            reward_discount_factor_gamma=0.99,
            entropy_penalty_coef=0.005)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=300))
test_rollout = Rollout(agent, gym.make("CartPole-v1"))

rewards = []
actor_loss = []
actor_utility = []
actor_kld = []
actor_entropy = []
critic_loss = []

for episode in range(1, 1001):
    roll_history = rollout.roll(steps=32, verbose=0, push_experience=True)
    agent_history = agent.fit(epochs=4, batch_size=32, verbose=0, reset_memory=True)

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    rewards.append(test_history["reward_sum"])
    actor_loss.append(np.mean(agent_history["actor_loss"]))
    actor_utility.append(np.mean(agent_history["actor_utility"]))
    actor_kld.append(np.mean(agent_history["actor_kld"]))
    actor_entropy.append(np.mean(agent_history["actor_entropy"]))
    critic_loss.append(np.mean(agent_history["critic_loss"]))

    print("\rEpisode {:>4} RWD {:>3.0f} ALOSS {: >7.4f} UTIL {: >7.4f} KKL {: >7.4f} ENTR {: >7.4f} CRIT {:.4f}".format(
        episode,
        np.mean(rewards[-10:]),
        np.mean(agent_history["actor_loss"]),
        np.mean(agent_history["actor_utility"]),
        np.mean(agent_history["actor_kld"])*1000,
        np.mean(agent_history["actor_entropy"]),
        np.mean(agent_history["critic_loss"])), end="")

    if episode % 10 == 0:
        print()

visual.plot_vectors([rewards, actor_loss, actor_utility, actor_kld, actor_entropy, critic_loss],
                    ["Reward", "Actor Loss", "Actor Utility", "Actor KLD", "Actor Entropy", "Critic Loss"],
                    window_size=10)
