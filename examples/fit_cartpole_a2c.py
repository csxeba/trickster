import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.experience import Experience
from trickster.utility import visual

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape, kernel_initializer="he_uniform"),
                    Dense(16, activation="relu", kernel_initializer="he_uniform"),
                    Dense(num_actions, activation="softmax", kernel_initializer="he_uniform")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape, kernel_initializer="he_uniform"),
                     Dense(16, activation="relu", kernel_initializer="he_uniform"),
                     Dense(1, activation="linear", kernel_initializer="he_uniform")])
critic.compile(loss="mse", optimizer=Adam(5e-4))

agent = A2C(actor,
            critic,
            action_space=env.action_space,
            memory=Experience(max_length=10000),
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.01)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gym.make("CartPole-v1"))

rewards = []
actor_loss = []
actor_utility = []
actor_entropy = []
critic_loss = []

for episode in range(1, 1001):
    episode_actor_loss = []
    episode_actor_utility = []
    episode_actor_entropy = []
    episode_critic_loss = []

    for update in range(32):
        rollout.roll(steps=2, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=32, verbose=0)
        episode_actor_loss.append(agent_history["actor_loss"])
        episode_actor_utility.append(agent_history["actor_utility"])
        episode_actor_entropy.append(agent_history["actor_entropy"])
        episode_critic_loss.append(agent_history["critic_loss"])

    test_history = test_rollout.rollout(verbose=0, push_experience=False)

    rewards.append(test_history["reward_sum"])
    actor_loss.append(sum(episode_actor_loss) / len(episode_actor_loss))
    actor_utility.append(sum(episode_actor_utility) / len(episode_actor_utility))
    actor_entropy.append(sum(episode_actor_entropy) / len(episode_actor_entropy))
    critic_loss.append(sum(episode_critic_loss) / len(episode_critic_loss))

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
