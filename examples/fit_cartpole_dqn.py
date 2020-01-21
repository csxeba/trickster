import gym

from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling
from trickster.model import mlp
from trickster.experience import replay_buffer

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = mlp.wide_mlp_critic(input_shape, num_actions, adam_lr=1e-3)

agent = DQN(ann,
            action_space=env.action_space,
            memory=replay_buffer.Experience(max_length=1024),
            epsilon=0.2,
            epsilon_decay=1.,
            epsilon_min=0.2,
            discount_factor_gamma=0.99,
            use_target_network=False)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.roll(steps=32, verbose=0, learning=True)
agent.epsilon_greedy.reset()

rollout.fit(epochs=300, updates_per_epoch=8, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True, render_every=0)
test_rollout.render(repeats=10)
