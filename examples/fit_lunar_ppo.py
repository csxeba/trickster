import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("LunarLanderContinuous-v2")

agent = PPO.from_environment(env)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, env)

rollout.fit(epochs=2000, rollouts_per_epoch=2, render_every=0, plot_curves=True)
