import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig
from trickster.utility import gym_utils

env = gym_utils.wrap(gym.make("Pendulum-v0"), reward_callback=lambda r: r / 6.)

agent = PPO.from_environment(env, discount_gamma=0.99, gae_lambda=0.97, entropy_beta=0.01, clip_epsilon=0.3)
agent.actor.optimizer.learning_rate = 1e-3

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=300))
rollout.fit(epochs=300, rollouts_per_epoch=1, render_every=0, plot_curves=True)
