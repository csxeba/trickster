import gym

from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

agent = DQN.from_environment(
    env=env,
    discount_gamma=0.99,
    epsilon=0.1,
    epsilon_decay=1.,
    epsilon_min=0.1,
    use_target_network=False)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.fit(epochs=300, updates_per_epoch=64, steps_per_update=1, update_batch_size=100,
            testing_rollout=test_rollout, plot_curves=True, render_every=0, warmup_buffer=True)
test_rollout.render(repeats=10)
