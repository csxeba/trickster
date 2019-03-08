import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.experience import Experience
from trickster.utility import visual

envs = [gym.make("CartPole-v1") for _ in range(4)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                    Dense(16, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(5e-4))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(1e-3))

agent = A2C(actor,
            critic,
            action_space=envs[0].action_space,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.005)

rollout = MultiRolling(agent.create_workers(4), envs, rollout_configs=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gym.make("CartPole-v1"))

rewards = []
actor_loss = []
actor_utility = []
actor_entropy = []
critic_loss = []

for episode in range(1, 301):
    episode_actor_loss = []
    episode_actor_utility = []
    episode_actor_entropy = []
    episode_critic_loss = []

    for update in range(32):
        rollout.roll(steps=4, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=-1, verbose=0)
        episode_actor_loss.append(agent_history["actor_loss"])
        episode_actor_utility.append(agent_history["actor_utility"])
        episode_actor_entropy.append(agent_history["actor_entropy"])
        episode_critic_loss.append(agent_history["critic_loss"])

    test_history = test_rollout.rollout(verbose=0, push_experience=False)

    rewards.append(test_history["reward_sum"])
    actor_loss.append(np.mean(episode_actor_loss))
    actor_utility.append(np.mean(episode_actor_utility))
    actor_entropy.append(np.mean(episode_actor_entropy))
    critic_loss.append(np.mean(episode_critic_loss))

    print("\rEpisode {:>4} RWD {:>5.2f} ACTR {:>7.4f} UTIL {:>7.4f} ENTR {:>7.4f} CRIT {:>7.4f}".format(
        episode,
        np.mean(rewards[-10:]),
        np.mean(actor_loss[-10:]),
        np.mean(actor_utility[-10:]),
        np.mean(actor_entropy[-10:]),
        np.mean(critic_loss[-10:])), end="")
    if episode % 10 == 0:
        print()

visual.plot_vectors([rewards, actor_loss, actor_utility, actor_entropy, critic_loss],
                    ["Reward", "Actor Loss", "Actor Utility", "Actor Entropy", "Critic Loss"],
                    smoothing_window_size=10)
