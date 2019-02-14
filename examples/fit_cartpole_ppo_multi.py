import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.advanced import PPO
from trickster.rollout import MultiRolling, Rollout, RolloutConfig
from trickster.experience import Experience
from trickster.utility import visual

NUM_PARALLEL_ENVS = 16
MAX_TIMESTEPS = 300
NUM_EPISODES = 1000
ROLL_TIMESTEPS = 32
FIT_EPOCHS = 4
FIT_BATCH_SIZE = 32
EXPERIENCE_SIZE = NUM_PARALLEL_ENVS * ROLL_TIMESTEPS
ACTOR_ADAM_LR = 1e-4
CRITIC_ADAM_LR = 1e-3
DISCOUNT_FACTOR_GAMMA = 0.99
ENTROPY_PENALTY_BETA = 0.005
SCALAR_SMOOTHING_WINDOW_SIZE = 10
ENV = "CartPole-v1"

envs = [gym.make(ENV) for _ in range(NUM_PARALLEL_ENVS)]
test_env = gym.make(ENV)

input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                    Dense(16, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(ACTOR_ADAM_LR))

critic = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(CRITIC_ADAM_LR))

agent = PPO(actor,
            critic,
            action_space=test_env.action_space,
            memory=Experience(max_length=EXPERIENCE_SIZE),
            reward_discount_factor_gamma=DISCOUNT_FACTOR_GAMMA,
            entropy_penalty_coef=ENTROPY_PENALTY_BETA)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=MAX_TIMESTEPS))
test_rollout = Rollout(agent, gym.make("CartPole-v1"))

rewards = []
actor_loss = []
actor_utility = []
actor_kld = []
actor_entropy = []
critic_loss = []

for episode in range(1, NUM_EPISODES+1):
    roll_history = rollout.roll(steps=ROLL_TIMESTEPS, verbose=0, push_experience=True)
    agent_history = agent.fit(epochs=FIT_EPOCHS, batch_size=FIT_BATCH_SIZE, verbose=0, reset_memory=False)

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    rewards.append(test_history["reward_sum"])
    actor_loss.append(np.mean(agent_history["actor_loss"]))
    actor_utility.append(np.mean(agent_history["actor_utility"]))
    actor_kld.append(np.mean(agent_history["actor_kld"]))
    actor_entropy.append(np.mean(agent_history["actor_entropy"]))
    critic_loss.append(np.mean(agent_history["critic_loss"]))

    print("\rEpisode {:>4} RWD {:>3.0f} ALOSS {: >7.4f} UTIL {: >7.4f} kKL {: >7.4f} ENTR {: >7.4f} CRIT {:.4f}".format(
        episode,
        np.mean(rewards[-SCALAR_SMOOTHING_WINDOW_SIZE:]),
        np.mean(actor_loss[-SCALAR_SMOOTHING_WINDOW_SIZE:]),
        np.mean(actor_utility[-SCALAR_SMOOTHING_WINDOW_SIZE:]),
        np.mean(actor_kld[-SCALAR_SMOOTHING_WINDOW_SIZE:])*1000,  # kiloKL :D
        np.mean(actor_entropy[-SCALAR_SMOOTHING_WINDOW_SIZE:]),
        np.mean(critic_loss[-SCALAR_SMOOTHING_WINDOW_SIZE:])), end="")

    if episode % 10 == 0:
        print()

visual.plot_vectors([rewards, actor_loss, actor_utility, actor_kld, actor_entropy, critic_loss],
                    ["Reward", "Actor Loss", "Actor Utility", "Actor KLD", "Actor Entropy", "Critic Loss"],
                    window_size=SCALAR_SMOOTHING_WINDOW_SIZE)
