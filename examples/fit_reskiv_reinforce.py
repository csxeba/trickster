from collections import deque

import numpy as np
import cv2
from keras.models import Model
from keras.layers import Conv2D, Input, LeakyReLU, Dense
from keras.optimizers import Adam

from trickster.agent import REINFORCE
from trickster.experience import Experience
from trickster.rollout import MultiTrajectory, RolloutConfig

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors
from grund.util.screen import CV2Screen


def preprocess(state):
    ds = cv2.resize(state, (0, 0), fx=0.5, fy=0.5)
    ds = ds / 255.
    return ds


TRAIN = True
WARMUP = 1
NUM_MOVES = 4
NUM_PARALLEL_ROLLOUTS = 4

MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
envs = [Reskiv(rcfg) for _ in range(NUM_PARALLEL_ROLLOUTS)]

canvas_shape, action_shape = envs[0].neurons_required

common_input = Input(shape=[32, 32, 3])

actor_stream = Conv2D(16, (3, 3), strides=(2, 2), padding="same")(common_input)  # 16
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(32, (3, 3), strides=(2, 2), padding="same")(actor_stream)  # 8
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(actor_stream)  # 4
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(128, (4, 4))(actor_stream)
actor_stream = LeakyReLU()(actor_stream)

actor_stream = Dense(32, activation="relu")(actor_stream)
action_probs = Dense(NUM_MOVES, activation="softmax")(actor_stream)

actor = Model(common_input, action_probs, name="Actor")
actor.compile(Adam(1e-3), "categorical_crossentropy")

agent = REINFORCE(actor, action_space=MOVES, memory=Experience(max_length=10000), discount_factor_gamma=0.99,
                  state_preprocessor=preprocess)

screen = CV2Screen(scale=2)
episode = 0
reward_memory = deque(maxlen=100)
critic_losses = deque(maxlen=50)
actor_losses = deque(maxlen=100)

rollout = MultiTrajectory(agent, envs, warmup_episodes=WARMUP, rollout_configs=RolloutConfig(max_steps=512, skipframes=4))
history = {"episode": 0}

while 1:
    rollout.reset()
    episode_actor_losses = []
    while not rollout.finished:
        history = rollout.roll(steps=4, verbose=0, learning_batch_size=32)
        reward_memory.append(history["reward_sum"])
        if "loss" in history:
            episode_actor_losses.append(history["loss"])
    actor.save("../artifacts/reskiv/reinforce_latest.h5")
    episode = history["episode"]

    if episode_actor_losses:
        actor_losses.append(np.mean(episode_actor_losses))
        print("\rEPISODE {} RRWD: {:.2f} LOSS {:.4f}".format(
            episode, np.mean(reward_memory), np.mean(actor_losses)), end="")
    else:
        print("EPISODE {}: WARMING UP...".format(episode))

    if episode >= 100 and episode % 100 == 0 and TRAIN:
        print(" Model dumplings...")
        model_path_template_pfx = "../artifacts/reskiv/reinforce_"
        if np.mean(reward_memory) > 3:
            model_path_template_sfx = "{}_r{:.2f}.h5".format(episode, np.mean(reward_memory))
        else:
            model_path_template_sfx = "latest.h5"
        actor.save(model_path_template_pfx + model_path_template_sfx)

    episode += 1
