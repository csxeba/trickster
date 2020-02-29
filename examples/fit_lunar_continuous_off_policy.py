import gym
import tensorflow as tf

from trickster.agent import DDPG, TD3, SAC
from trickster.rollout import Trajectory, MultiRolling

ENV_NAME = "LunarLanderContinuous-v2"
ALGO = "SAC"
TRAJECTORY_MAX_STEPS = 400
STEPS_PER_UPDATE = 1
UPDATES_PER_EPOCH = 32
EPOCHS = 1000
UPDATE_BATCH_SIZE = 64
NUM_ENVS = 8

envs = [gym.make(ENV_NAME) for _ in range(NUM_ENVS)]
test_env = gym.make(ENV_NAME)

algo = {"DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC}[ALGO]

agent = algo.from_environment(envs[0])
agent.actor.optimizer = tf.keras.optimizers.RMSprop(3e-4)

rollout = MultiRolling(agent, envs, TRAJECTORY_MAX_STEPS)
test_rollout = Trajectory(agent, test_env, TRAJECTORY_MAX_STEPS)

rollout.fit(epochs=EPOCHS,
            updates_per_epoch=UPDATES_PER_EPOCH,
            steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE,
            testing_rollout=test_rollout,
            warmup_buffer=True)

test_rollout.render(repeats=10, verbose=0)
