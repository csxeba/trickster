import gym

from trickster.agent import REINFORCE, PPO
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("Pendulum-v0")

agent = PPO.from_environment(env)
agent.actor.optimizer.learning_rate = 3e-4
agent.critic.optimizer.learning_rate = 3e-4

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))
rollout.fit(epochs=2000, rollouts_per_epoch=1, render_every=0, plot_curves=True)

