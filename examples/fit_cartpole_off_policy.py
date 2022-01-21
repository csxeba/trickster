import gym

from trickster.agent import DQN, DoubleDQN
from trickster.rollout import Trajectory, MultiRolling
from trickster import callbacks

ENV_NAME = "CartPole-v1"
ALGO = "DQN"
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

rollout = MultiRolling(agent, envs, TRAJECTORY_MAX_STEPS)
test_rollout = Trajectory(agent, test_env, TRAJECTORY_MAX_STEPS)

rollout.fit(epochs=EPOCHS,
            updates_per_epoch=UPDATES_PER_EPOCH,
            steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE,
            warmup_buffer=True,
            callbacks=[callbacks.TrajectoryEvaluator(testing_rollout=test_rollout, repeats=4),
                       callbacks.ProgressPrinter(rollout.progress_keys)])

test_rollout.render(repeats=10)
