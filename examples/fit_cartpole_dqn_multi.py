import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import DQN
from trickster.experience import Experience
from trickster.rollout import MultiRolling, RolloutConfig, Trajectory
from trickster.utility import visual

envs = [gym.make("CartPole-v1") for _ in range(4)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

qnet = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                   Dense(16, activation="relu"),
                   Dense(num_actions, activation="linear")])
qnet.compile(loss="mse", optimizer=Adam(1e-3))

agent = DQN(qnet,
            action_space=2,
            memory=Experience(max_length=10000),
            epsilon=1.,
            discount_factor_gamma=0.98,
            use_target_network=True)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gym.make("CartPole-v1"))

rewards = []
losses = []

for episode in range(1, 501):
    episode_losses = []

    for update in range(32):
        rollout.roll(steps=4, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=32, verbose=0)
        episode_losses.append(agent_history["loss"])

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
    rewards.append(test_history["reward_sum"])
    losses.append(np.mean(episode_losses))
    print("\rEpisode {:>4} RWD {:>3.0f} LOSS {:.4f} EPS {:.2%}".format(
        episode, np.mean(rewards[-10:]), np.mean(losses[-10:]), agent.epsilon), end="")

    agent.epsilon *= 0.992
    agent.epsilon = max(agent.epsilon, 0.01)
    if episode % 10 == 0:
        agent.push_weights()
        print(" Pushed weights to target net!")

visual.plot_vectors([rewards, losses], ["Reward", "Loss"], smoothing_window_size=10)
