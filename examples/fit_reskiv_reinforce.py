from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, LeakyReLU, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

from trickster import REINFORCE, Experience, MultiRollout, RolloutConfig

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors
from grund.util.screen import CV2Screen

TRAIN = True
WARMUP = 1
NUM_MOVES = 4
NUM_PARALLEL_ROLLOUTS = 4

MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
envs = [Reskiv(rcfg) for _ in range(NUM_PARALLEL_ROLLOUTS)]

canvas_shape, action_shape = envs[0].neurons_required

common_input = Input(shape=[64, 64, 3])

actor_stream = Conv2D(16, (3, 3), strides=(2, 2), padding="same")(common_input)  # 32
actor_stream = BatchNormalization()(LeakyReLU()(actor_stream))
actor_stream = Conv2D(32, (3, 3), strides=(2, 2), padding="same")(actor_stream)  # 16
actor_stream = BatchNormalization()(LeakyReLU()(actor_stream))
actor_stream = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(actor_stream)  # 8
actor_stream = BatchNormalization()(LeakyReLU()(actor_stream))
actor_stream = Conv2D(NUM_MOVES, (1, 1))(actor_stream)
actor_stream = GlobalAveragePooling2D()(actor_stream)
# actor_stream = Flatten()(actor_stream)
action_probs = Activation("softmax")(actor_stream)

actor = Model(common_input, action_probs, name="Actor")
actor.compile(Adam(1e-4), "categorical_crossentropy")

agent = REINFORCE(actor, actions=MOVES, memory=Experience(max_length=10000), reward_discount_factor=0.99,
                  state_preprocessor=lambda state: state / 255.)

screen = CV2Screen(scale=2)
episode = 0
reward_memory = deque(maxlen=100)
critic_losses = deque(maxlen=50)
actor_losses = deque(maxlen=100)

rollout = MultiRollout(agent, envs, warmup_episodes=WARMUP, rollout_configs=RolloutConfig(max_steps=512))
history = {"episode": 0}

while 1:
    rollout.reset()
    episode_actor_losses = []
    while not rollout.finished:
        history = rollout.roll(steps=4, verbose=0, learning_batch_size=32)
        reward_memory.append(history["reward_sum"])
        if "loss" in history:
            episode_actor_losses.append(history["loss"])

    episode = history["episode"]

    if episode_actor_losses:
        actor_losses.append(np.mean(episode_actor_losses))
        print("\rEPISODE {} RRWD: {:.2f} LOSS {:.4f}".format(
            episode, np.mean(reward_memory), np.mean(actor_losses)), end="")
    else:
        print("EPISODE {}: WARMING UP...".format(episode))

    if episode >= 100 and episode % 100 == 0 and TRAIN:
        print(" Model dumplings...")
        model_path_template_pfx = "../models/reskiv/reinforce_"
        if np.mean(reward_memory) > 3:
            model_path_template_sfx = "{}_r{:.2f}.h5".format(episode, np.mean(reward_memory))
        else:
            model_path_template_sfx = "latest.h5"
        actor.save(model_path_template_pfx + model_path_template_sfx)

    episode += 1
