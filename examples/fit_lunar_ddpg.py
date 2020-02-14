import gym

from trickster.agent import SAC
from trickster.rollout import Trajectory, Rolling, RolloutConfig

env = gym.make("LunarLanderContinuous-v2")
test_env = gym.make("LunarLanderContinuous-v2")

agent = SAC.from_environment(env, entropy_beta=0.1)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env)

agent.actor.optimizer.learning_rate = 1e-3
agent.critic.optimizer.learning_rate = 1e-3

rollout.fit(epochs=1000, updates_per_epoch=64, steps_per_update=1, update_batch_size=100,
            testing_rollout=test_rollout, render_every=0, plot_curves=True, warmup_buffer=10000)
