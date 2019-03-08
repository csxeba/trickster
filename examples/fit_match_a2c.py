from keras.models import Sequential
from keras.layers import Dense

from grund.match import MatchConfig, Match

from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.utility import history, visual

cfg = MatchConfig(canvas_size=(100, 100), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

envs = [Match(cfg) for _ in range(4)]
test_env = Match(cfg)

actor = Sequential([
    Dense(32, activation="relu", input_shape=test_env.observation_space.shape),
    Dense(32, activation="relu"),
    Dense(test_env.action_space.n, activation="softmax")
])
actor.compile("adam", "categorical_crossentropy")

critic = Sequential([
    Dense(32, activation="relu", input_shape=test_env.observation_space.shape),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])
critic.compile("adam", "mse")

agent = A2C(actor, critic, test_env.action_space, absolute_memory_limit=10000, entropy_penalty_coef=0.05)

rcfg = RolloutConfig(max_steps=1024, skipframes=2)
training_rollout = MultiRolling(agent.create_workers(4), envs, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

episode = 1

while 1:

    logger = history.History("reward", "actor_loss", "actor_utility", "actor_utility_std", "actor_entropy", "critic_loss")

    for logging_round in range(300):

        for update in range(32):
            training_rollout.roll(steps=32, verbose=0, push_experience=True)
            agent_history = agent.fit(batch_size=-1, verbose=0)
            logger.buffer(**agent_history)

        for _ in range(5):
            test_history = testing_rollout.rollout(verbose=0, push_experience=False, render=False)
            logger.buffer(reward=test_history["reward_sum"])

        logger.push_buffer()
        logger.print(average_last=10,
                     templates={"reward": "RWD {: >5.2f}",
                                "actor_loss": "LOSS {: >6.4f}",
                                "actor_utility": "UTIL {: >6.4f}",
                                "actor_utility_std": "STD {:>6.4f}",
                                "actor_entropy": "ENTR {:>6.4f}",
                                "critic_loss": "CRIT {:>6.4f}"},
                     prefix="Episode {} ".format(episode))

        episode += 1

    # visual.plot_history(logger, smoothing_window_size=10)
