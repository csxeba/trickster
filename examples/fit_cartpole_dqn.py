from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling
from trickster.experience import Experience
from trickster.model import mlp
from trickster.utility import gymic

env = gymic.rwd_scaled_env("CartPole-v1")
test_env = gymic.rwd_scaled_env("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = mlp.wide_mlp_critic(input_shape, num_actions, adam_lr=1e-3)

agent = DQN(ann,
            action_space=env.action_space,
            memory=Experience(max_length=10000),
            epsilon=1.,
            epsilon_decay=0.99995,
            epsilon_min=0.1,
            discount_factor_gamma=0.98,
            use_target_network=True)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env)

rollout.roll(steps=1024, verbose=0, push_experience=True)

rollout.fit(episodes=500, updates_per_episode=32, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=10)
