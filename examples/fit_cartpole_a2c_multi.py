from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory
from trickster.utility import gymic
from trickster.model import mlp

envs = [gymic.rwd_scaled_env() for _ in range(8)]
input_shape = envs[0].observation_space.shape
num_actions = envs[0].action_space.n

actor, critic = mlp.wide_pg_actor_critic(input_shape, num_actions, critic_lr=5e-4)

agent = A2C(actor,
            critic,
            action_space=envs[0].action_space,
            discount_factor_gamma=0.98,
            entropy_penalty_coef=0.05)

rollout = MultiRolling(agent, envs)
test_rollout = Trajectory(agent, gymic.rwd_scaled_env())

rollout.fit(episodes=300, updates_per_episode=128, steps_per_update=1, testing_rollout=test_rollout)
test_rollout.render()
