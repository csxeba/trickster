from grund.match import MatchConfig, Match

from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.experience import Experience
from trickster.utility import history, visual
from trickster.model import mlp

cfg = MatchConfig(canvas_size=(128, 128), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

env = Match(cfg)
test_env = Match(cfg)

ann = mlp.wide_dueling_q_network(env.observation_space.shape, env.action_space.n, adam_lr=1e-4)

experience = Experience(10000)
agent = DoubleDQN(ann, env.action_space, experience,
                  epsilon=1., epsilon_decay=1., epsilon_min=0.1)

rcfg = RolloutConfig(max_steps=1024, skipframes=2)
training_rollout = Rolling(agent, env, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

print("Filling experience...")
while experience.N < 10000:
    training_rollout.roll(steps=32, verbose=0, push_experience=True)
    print(f"\r{experience.N/10000:.2%} 10000/{experience.N}", end="")
print()
agent.epsilon_decay = 0.99995

logger = history.History("reward_sum", *agent.history_keys, "epsilon")

for episode in range(1, 501):

    for update in range(32):
        training_rollout.roll(steps=32, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=1024, verbose=0, polyak_tau=0.1)
        logger.buffer(**agent_history)

    for _ in range(3):
        test_history = testing_rollout.rollout(verbose=0, push_experience=False, render=False)
        logger.buffer(reward_sum=test_history["reward_sum"])

    logger.push_buffer()
    logger.record(epsilon=agent.epsilon)

    logger.print(average_last=10, return_carriege=True, prefix="Episode {}".format(episode))

visual.plot_history(logger, smoothing_window_size=10, skip_first=0, show=True)
