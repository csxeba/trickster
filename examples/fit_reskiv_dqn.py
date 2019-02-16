from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam

from trickster.agent import DQN
from trickster.experience import Experience
from trickster.rollout import RolloutConfig, Trajectory

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors
from grund.util.screen import CV2Screen

TRAIN = True
WARMUP = 10
NUM_MOVES = 4
NUM_PARALLEL_ROLLOUTS = 4

MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
env = Reskiv(rcfg)

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

screen = CV2Screen(scale=2)
episode = 0
reward_memory = deque(maxlen=100)
losses = deque(maxlen=100)

rollout = Trajectory(agent, env, config=RolloutConfig(max_steps=512))
history = {"episode": 0}

print("Doing {} warmup rollouts...".format(WARMUP))
agent.epsilon = 1.
baseline_reward = 0.
for w in range(1, WARMUP+1):
    print("\r{}/{}".format(WARMUP, w), end="")
    rwd = rollout.rollout(verbose=0, learning_batch_size=0)
    baseline_reward += rwd
baseline_reward /= WARMUP
print("\nBaseline reward: {:.4f}".format(baseline_reward))

last_best_running_reward = 0.

while 1:
    episode += 1
    rollout._reset()
    reward_sum = 0.
    episode_losses = []
    while not rollout.done:
        history = rollout.roll(steps=1, verbose=0, learning_batch_size=4)
        reward_sum += history["reward_sum"]
        episode_losses.append(history["loss"])

    reward_memory.append(reward_sum)
    losses.append(np.mean(episode_losses))
    running_reward = np.mean(reward_memory)

    print("\rEP {}: RWD {:.0f} RRWD {:.4f} LOSS {:.4f} EPS: {:.2%}"
          .format(episode, reward_memory[-1], running_reward, np.mean(losses), agent.epsilon),
          end="")

    if episode % 100 == 0:
        critic.save("../models/reskiv/critic_latest.h5")
        print(" Dumping models...")
        if running_reward > 5. and running_reward > last_best_running_reward:
            critic.save("../models/reskiv/critic_{}_r{:.4f}.h5".format(episode, running_reward))

    last_best_running_reward = max(last_best_running_reward, running_reward)

    if agent.epsilon > 0.1:
        agent.epsilon *= 0.999
    else:
        agent.epsilon = 0.01
