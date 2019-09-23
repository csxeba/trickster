from grund.match import MatchConfig, Match

from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.experience import Experience
from trickster.model import mlp

cfg = MatchConfig(canvas_size=(128, 128), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

env = Match(cfg)
test_env = Match(cfg)

ann = mlp.wide_dueling_q_network(env.observation_space.shape, env.action_space.n, adam_lr=1e-4)

experience = Experience(100000)
agent = DoubleDQN(ann, env.action_space, experience,
                  epsilon=1., epsilon_decay=1., epsilon_min=0.1)

rcfg = RolloutConfig(max_steps=512, skipframes=2)
training_rollout = Rolling(agent, env, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

print("Filling experience...")
while experience.N < 100000:
    training_rollout.roll(steps=100, verbose=0, push_experience=True)
    print(f"\r{experience.N/100000:.2%} 100000/{experience.N}", end="")
print()
agent.epsilon_decay = 0.99999

training_rollout.fit(episodes=1000, updates_per_episode=256, step_per_update=2, update_batch_size=256,
                     testing_rollout=testing_rollout, render_every=100)
