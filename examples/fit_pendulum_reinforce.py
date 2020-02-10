import gym

from trickster.agent import A2C
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("Pendulum-v0")

agent = A2C.from_environment(env, discount_gamma=0.99, gae_lambda=0.97, entropy_beta=0.01)
rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))

agent.update_actor = 0
agent.critic.optimizer.learning_rate = 1e-1
rollout.fit(epochs=20, rollouts_per_epoch=1, render_every=0, plot_curves=False)
agent.critic.optimizer.learning_rate = 1e-2
rollout.fit(epochs=20, rollouts_per_epoch=1, render_every=0, plot_curves=False)

agent.update_actor = 1
rollout.fit(epochs=300, rollouts_per_epoch=1, render_every=0, plot_curves=True)

