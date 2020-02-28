import gym

from trickster.agent import DQN, DoubleDQN
from trickster.rollout import Trajectory, RolloutConfig, MultiRolling

ENV_NAME = "CartPole-v1"
ALGO = "DoubleDQN"
NUM_ENVS = 4
TRAJECTORY_MAX_STEPS = 200
STEPS_PER_UPDATE = 1
UPDATES_PER_EPOCH = 64
EPOCHS = 200
UPDATE_BATCH_SIZE = 100

envs = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
test_env = gym.make(ENV_NAME)

algo = {"DQN": DQN,
        "DoubleDQN": DoubleDQN}[ALGO]

agent = algo.from_environment(envs[0])

cfg = RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS)

rollout = MultiRolling(agent, envs, rollout_configs=cfg)
test_rollout = Trajectory(agent, test_env, config=cfg)

rollout.fit(epochs=EPOCHS, updates_per_epoch=UPDATES_PER_EPOCH, steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE,
            testing_rollout=test_rollout, plot_curves=True, render_every=0, warmup_buffer=True)

test_rollout.render(repeats=10)
