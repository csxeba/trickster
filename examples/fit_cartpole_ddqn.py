import gym

from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

agent = DoubleDQN.from_environment(
    env,
    discount_gamma=0.98,
    epsilon=1.,
    epsilon_decay=0.999,
    epsilon_min=0.1)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.roll(steps=1000, verbose=0, learning=True)  # Collect some random trajectories
agent.epsilon_greedy.reset()

rollout.fit(epochs=300, updates_per_epoch=64, steps_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True, render_every=0)
test_rollout.render(repeats=10)
