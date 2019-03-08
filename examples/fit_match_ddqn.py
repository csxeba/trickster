from collections import deque

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from grund.match import MatchConfig, Match

from trickster.agent import DoubleDQN
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.experience import Experience

cfg = MatchConfig(canvas_size=(100, 100), players_per_side=2,
                  learning_type=MatchConfig.LEARNING_TYPE_SINGLE_AGENT,
                  observation_type=MatchConfig.OBSERVATION_TYPE_VECTOR)

env = Match(cfg)
test_env = Match(cfg)

ann = Sequential([
    Dense(64, activation="relu", input_shape=env.observation_space.shape),
    Dense(env.action_space.n, activation="linear")
])
ann.compile("adam", "mse")

experience = Experience(10000)
agent = DoubleDQN(ann, env.action_space, experience)

rcfg = RolloutConfig(max_steps=1024, skipframes=2)
training_rollout = Rolling(agent, env, rcfg)
testing_rollout = Trajectory(agent, test_env, rcfg)

print("Filling experience...")
while experience.N < 10000:
    training_rollout.roll(steps=32, verbose=0, push_experience=True)
    print(f"\r{experience.N/10000:.2%} 10000/{experience.N}", end="")
print()

episode = 1
reward_memory = deque(maxlen=10)
while 1:
    loss = 0.
    for update in range(32):
        training_rollout.roll(steps=32, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=1024, verbose=0, update_target=False)
        loss += agent_history["loss"]
    loss /= 32

    rewards = 0
    for _ in range(5):
        test_history = testing_rollout.rollout(verbose=0, push_experience=False, render=False)
        rewards += test_history["reward_sum"]
    rewards /= 5
    reward_memory.append(rewards)

    print(f"\rEpisode {episode} RWD {np.mean(reward_memory): >5.2f} LOSS {loss:.4f}", end="")

    agent.meld_weights(mix_in_ratio=0.1)

    episode += 1
