from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.experience import Experience
from trickster.model import mlp
from trickster.utility import gym_utils

env = gym_utils.rwd_scaled_env("CartPole-v1")
test_env = gym_utils.rwd_scaled_env("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = mlp.wide_mlp_critic(input_shape, num_actions, adam_lr=1e-3)

agent = DoubleDQN(ann,
                  action_space=env.action_space,
                  memory=Experience(max_length=10000),
                  epsilon=1.,
                  epsilon_decay=0.99995,
                  epsilon_min=0.1,
                  discount_factor_gamma=0.98)


rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, test_env)

rollout.fit(epochs=500, updates_per_epoch=32, step_per_update=2, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True)
test_rollout.render(repeats=10)
