import gym

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("Pendulum-v0")

agent = REINFORCE.from_environment(env, discount_gamma=0.99)
agent.actor.optimizer.learning_rate = 1e-4

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))
rollout.fit(epochs=300, rollouts_per_epoch=1, render_every=0, plot_curves=True)

