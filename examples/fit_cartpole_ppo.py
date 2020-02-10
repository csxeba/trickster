import gym

from trickster.agent import PPO
from trickster.rollout import Trajectory, RolloutConfig

env = gym.make("CartPole-v1")

agent = PPO.from_environment(
    env,
    discount_gamma=0.99,
    gae_lambda=0.97,
    entropy_beta=0.001,
    clip_epsilon=0.3,
    target_kl_divergence=0.01,
    actor_updates=80,
    critic_updates=80)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=200))

rollout.fit(epochs=300, rollouts_per_epoch=100, update_batch_size=32, plot_curves=True, render_every=0)
rollout.render(repeats=100)
