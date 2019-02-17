from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam

from trickster.agent import DQN
from trickster.experience import Experience
from trickster.rollout import RolloutConfig, Trajectory, Rolling

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors

TRAIN = True
NUM_MOVES = 4
NUM_PARALLEL_ROLLOUTS = 4

MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
env = Reskiv(rcfg)
test_env = Reskiv(rcfg)

canvas_shape, action_shape = env.neurons_required

actor_input = Input(shape=[64, 64, 3], name="actor_input")
critic_input = Input(shape=[64, 64, 3], name="critic_input")

critic_stream = Flatten()(critic_input)
critic_stream = Dense(64, activation="tanh")(critic_stream)
critic_stream = BatchNormalization()(critic_stream)
critic_stream = Dense(32, activation="tanh")(critic_stream)
critic_stream = BatchNormalization()(critic_stream)
value_estimate = Dense(NUM_MOVES, activation="softmax")(critic_stream)

critic = Model(critic_input, value_estimate, name="Critic")
critic.compile(Adam(5e-4), "mse")

agent = DQN(critic, action_space=MOVES, memory=Experience(max_length=10000), discount_factor_gamma=0.99, epsilon=0.7,
            state_preprocessor=lambda state: state / 255. - 0.5)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=512, skipframes=2))
test_rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=512, skipframes=2))

episode = 0
reward_memory = deque(maxlen=10)
losses = deque(maxlen=10)

while 1:
    episode += 1

    episode_losses = []
    for update in range(32):
        rollout.roll(steps=4, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=32, verbose=0)
        episode_losses.append(agent_history["loss"])

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    reward_memory.append(test_history["reward_sum"])
    losses.append(np.mean(episode_losses))

    print("\rEP {:>4} STEP {:>4} RWD {: >6.1f} LOSS {: >7.4f} EPS: {:.2%}".format(
        episode,
        test_history["steps"],
        np.mean(reward_memory),
        np.mean(losses),
        agent.epsilon), end="")

    if agent.epsilon > 0.1:
        agent.epsilon *= 0.999
    else:
        agent.epsilon = 0.01
