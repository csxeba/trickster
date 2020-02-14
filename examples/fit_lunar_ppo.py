import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("LunarLanderContinuous-v2")

agent = PPO.from_environment(env, actor_updates=20, critic_updates=20, entropy_beta=-0.1)
agent.actor.optimizer.learning_rate = 1e-4

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))

rollout.fit(epochs=2000, rollouts_per_epoch=1, render_every=0, plot_curves=True)
