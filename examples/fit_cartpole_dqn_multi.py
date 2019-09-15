from trickster.agent import DQN
from trickster.experience import Experience
from trickster.rollout import MultiRolling, RolloutConfig, Trajectory
from trickster.utility import gymic
from trickster.model import mlp

envs = [gymic.rwd_scaled_env("CartPole-v1") for _ in range(8)]
test_env = gymic.rwd_scaled_env("CartPole-v1")

input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

ann = mlp.wide_mlp_critic_network(input_shape, num_actions, adam_lr=1e-4)

agent = DQN(ann,
            action_space=2,
            memory=Experience(max_length=10000),
            epsilon=1.,
            epsilon_decay=0.99995,
            epsilon_min=0.1,
            discount_factor_gamma=0.98,
            use_target_network=True)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env)

rollout.fit(episodes=500, updates_per_episode=128, steps_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render()
