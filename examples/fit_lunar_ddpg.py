import gym

from trickster.agent import DDPG
from trickster.rollout import Trajectory, Rolling, RolloutConfig

env = gym.make("LunarLanderContinuous-v2")
test_env = gym.make("LunarLanderContinuous-v2")

agent = DDPG.from_environment(env,
                              discount_gamma=0.99,
                              action_noise_sigma=0.1,
                              action_noise_sigma_decay=1.)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env)

rollout.fit(epochs=2000, updates_per_epoch=64, steps_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, render_every=0, plot_curves=True, warmup_buffer=True)
