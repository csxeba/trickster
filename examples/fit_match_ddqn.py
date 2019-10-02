from tensorflow import keras
from grund.match import MatchConfig, Match

from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.model import mlp


def ql2(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_pred), axis=-1)


cfg = MatchConfig(canvas_size=(128, 128), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR,
                  frameskip=4)

env = Match(cfg)
test_env = Match(cfg)

ann = mlp.wide_dueling_q_network(env.observation_space.shape, env.action_space.n, adam_lr=1e-4, batch_norm=True)
ann.add_loss(ql2)

agent = DoubleDQN(ann,
                  env.action_space,
                  epsilon=1.,
                  epsilon_decay=1.,
                  epsilon_min=0.1)

rcfg = RolloutConfig(max_steps=256)
training_rollout = Rolling(agent, env, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

print("Filling experience...")
while agent.memory.N < agent.memory.max_length:
    training_rollout.roll(steps=100, verbose=0, push_experience=True)
    print(f"\r{agent.memory.N/agent.memory.max_length:.2%} {agent.memory.max_length}/{agent.memory.N}", end="")
print()

agent.epsilon_decay = 0.99999

training_rollout.fit(episodes=2000, updates_per_episode=32, step_per_update=1, update_batch_size=64,
                     testing_rollout=testing_rollout, render_every=100)
testing_rollout.render(repeats=100)
