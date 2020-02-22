import gym

from trickster.agent import DQN, DoubleDQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling

ENV_NAME = "CartPole-v1"
ALGO = "DoubleDQN"
TRAJECTORY_MAX_STEPS = 200
STEPS_PER_UPDATE = 1
UPDATES_PER_EPOCH = 64
EPOCHS = 200
UPDATE_BATCH_SIZE = 100

env = gym.make(ENV_NAME)
test_env = gym.make(ENV_NAME)

algo = {"DQN": DQN,
        "DoubleDQN": DoubleDQN}[ALGO]

agent = algo.from_environment(env)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS))

rollout.fit(epochs=EPOCHS, updates_per_epoch=UPDATES_PER_EPOCH, steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE,
            testing_rollout=test_rollout, plot_curves=True, render_every=0, warmup_buffer=True)

test_rollout.render(repeats=10)
