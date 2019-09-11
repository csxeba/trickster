import os

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras

from grund.match import MatchConfig, Match

from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.utility import history, visual

cfg = MatchConfig(canvas_size=(100, 100), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

envs = [Match(cfg) for _ in range(4)]
test_env = Match(cfg)

actor = Sequential([
    Dense(400, activation="relu", input_shape=test_env.observation_space.shape),
    Dense(300, activation="relu"),
    Dense(test_env.action_space.n, activation="softmax")
])
actor.compile(keras.optimizers.Adam(1e-4), "categorical_crossentropy")

critic = Sequential([
    Dense(400, activation="relu", input_shape=test_env.observation_space.shape),
    Dense(300, activation="relu"),
    Dense(1, activation="linear")
])
critic.compile(keras.optimizers.Adam(1e-4), "mse")

agent = A2C(actor, critic, test_env.action_space, entropy_penalty_coef=0.05)

rcfg = RolloutConfig(max_steps=512, skipframes=2)
training_rollout = MultiRolling(agent.create_workers(4), envs, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

episode = 0
logger = history.History("reward_sum", *agent.HST_KEYS)

for episode in range(1, 1001):

    for update in range(10):
        training_rollout.roll(steps=64, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=-1, verbose=0)
        logger.buffer(**agent_history)

    for _ in range(3):
        test_history = testing_rollout.rollout(verbose=0, push_experience=False, render=False)
        logger.buffer(reward_sum=test_history["reward_sum"])

    logger.push_buffer()
    logger.print(average_last=100,
                 prefix="Episode {} ".format(episode))

    if episode % 100 == 0:
        print()

os.makedirs("Match", exist_ok=True)
template = "Match/match_a2c_{}_e{}_r{}.h5".format("{}", episode, logger.last()["reward_sum"])
actor.save_weights(template.format("actor"))
critic.save_weights(template.format("critic"))
visual.plot_history(logger, smoothing_window_size=100, show=False)
plt.tight_layout()
plt.savefig("Match/match_a2c_e{}_r{}.png".format(episode, logger.last()["reward_sum"]))
plt.clf()
