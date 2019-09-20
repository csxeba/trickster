from grund.match import MatchConfig, Match

from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.model import mlp

cfg = MatchConfig(canvas_size=(100, 100), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

envs = [Match(cfg) for _ in range(8)]
test_env = Match(cfg)

actor, critic = mlp.wide_pg_actor_critic(envs[0].observation_space.shape,
                                         envs[0].action_space.n,
                                         actor_lr=1e-4,
                                         critic_lr=1e-4)

agent = A2C(actor, critic, test_env.action_space, entropy_penalty_coef=1e-3)

rcfg = RolloutConfig(max_steps=512, skipframes=2)

training_rollout = MultiRolling(agent, envs, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

training_rollout.fit(episodes=1000, updates_per_episode=128, steps_per_update=1, testing_rollout=testing_rollout)
testing_rollout.render(repeats=10)
