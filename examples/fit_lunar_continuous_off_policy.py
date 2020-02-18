import gym

from trickster.agent import DDPG, TD3, SAC
from trickster.rollout import Trajectory, Rolling, RolloutConfig

ENV_NAME = "LunarLanderContinuous-v2"
ALGO = "TD3"
TRAJECTORY_MAX_STEPS = 300
STEPS_PER_UPDATE = 1
UPDATES_PER_EPOCH = 32
EPOCHS = 200
UPDATE_BATCH_SIZE = 64

env = gym.make(ENV_NAME)
test_env = gym.make(ENV_NAME)

algo = {"DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC}[ALGO]

agent = algo.from_environment(env, discount_gamma=0.8)
agent.actor.optimizer.learning_rate = 1e-4

cfg = RolloutConfig(max_steps=TRAJECTORY_MAX_STEPS)

rollout = Rolling(agent, env, cfg)
test_rollout = Trajectory(agent, test_env, cfg)

rollout.fit(epochs=EPOCHS, updates_per_epoch=UPDATES_PER_EPOCH, steps_per_update=STEPS_PER_UPDATE,
            update_batch_size=UPDATE_BATCH_SIZE, testing_rollout=test_rollout,
            render_every=100, plot_curves=True, warmup_buffer=10000)
test_rollout.render(repeats=10, verbose=0)
