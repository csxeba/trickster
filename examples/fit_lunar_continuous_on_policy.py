import gym

from trickster.agent import REINFORCE, A2C, PPO
from trickster.rollout import Trajectory
from trickster import callbacks

ENV_NAME = "LunarLanderContinuous-v2"
ALGO = "PPO"
TRAJECTORY_MAX_STEPS = 100
EPOCHS = 1000
ROLLOUTS_PER_EPOCH = 4

env = gym.make(ENV_NAME)

algo = {"REINFORCE": REINFORCE,
        "A2C": A2C,
        "PPO": PPO}[ALGO]

agent = algo.from_environment(env)
rollout = Trajectory(agent, env, TRAJECTORY_MAX_STEPS)

cbs = [callbacks.ProgressPrinter(keys=rollout.history_keys),
       callbacks.TrajectoryRenderer(testing_rollout=rollout),
       callbacks.TensorBoard(experiment_name=rollout.experiment_name)]

rollout.fit(epochs=EPOCHS, rollouts_per_epoch=ROLLOUTS_PER_EPOCH, callbacks=cbs)
rollout.render(repeats=100)
