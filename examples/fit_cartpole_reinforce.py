import numpy as np

import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory, RolloutConfig
from trickster.utility import visual, history

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(16, activation="relu", input_shape=input_shape),
                     Dense(16, activation="relu"),
                     Dense(num_actions, activation="softmax")])
policy.compile(loss="categorical_crossentropy", optimizer=Adam(5e-3))
agent = REINFORCE(policy, action_space=num_actions)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))

hst = history.History("reward_sum", "loss", "entropy")

for episode in range(1, 501):
    rollout_history = rollout.rollout(verbose=0, push_experience=True)
    agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)

    hst.record(reward_sum=rollout_history["reward_sum"], **agent_history)
    hst.print(average_last=100, templates={
        "reward_sum": "{:.4f}", "loss": "{:.4f}", "entropy": "{:.4f}", "epsilon": "{:.2%}"
    }, return_carriege=True, prefix="Episode {:>4}".format(episode))
    if episode % 100 == 0:
        print()

visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
