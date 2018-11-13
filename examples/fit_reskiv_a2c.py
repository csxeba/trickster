from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, LeakyReLU, BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

from trickster import A2C, Experience, MultiRollout, RolloutConfig

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

critic_stream = Conv2D(16, (3, 3), strides=(2, 2), padding="same")(common_input)  # 32
critic_stream = BatchNormalization()(LeakyReLU()(critic_stream))
critic_stream = Conv2D(32, (3, 3), strides=(2, 2), padding="same")(critic_stream)  # 16
critic_stream = BatchNormalization()(LeakyReLU()(critic_stream))
critic_stream = Conv2D(64, (3, 3), strides=(2, 2), padding="same")(critic_stream)  # 8
critic_stream = BatchNormalization()(LeakyReLU()(critic_stream))
critic_stream = Conv2D(1, (1, 1), padding="valid")(critic_stream)
value_estimate = GlobalAveragePooling2D()(critic_stream)
# value_estimate = Flatten()(critic_stream)

actor = Model(common_input, action_probs, name="Actor")
actor.compile(Adam(1e-4), "categorical_crossentropy")
critic = Model(common_input, value_estimate, name="Critic")
critic.compile(Adam(5e-4), "mse")

agent = A2C(actor, critic, actions=MOVES, memory=Experience(max_length=10000), reward_discount_factor=0.99,
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
    episode_critic_losses = []
    while not rollout.finished:
        history = rollout.roll(steps=4, verbose=0, learning_batch_size=32)
        reward_memory.append(history["reward_sum"])
        if "actor_loss" in history:
            episode_actor_losses.append(history["actor_loss"])
        if "critic_loss" in history:
            episode_critic_losses.append(history["critic_loss"])

    episode = history["episode"]

    if episode_actor_losses and episode_critic_losses:
        actor_losses.append(np.mean(episode_actor_losses))
        critic_losses.append(np.mean(episode_critic_losses))
        print("\rEPISODE {} RRWD: {:.2f} ACTR {:.4f} CRIT {:.4f}".format(
            episode, np.mean(reward_memory), np.mean(actor_losses), np.mean(critic_losses)), end="")
    else:
        print("EPISODE {}: WARMING UP...".format(episode))

    if episode >= 100 and episode % 100 == 0 and TRAIN:
        print(" Model dumplings...")
        model_path_template_pfx = "../models/reskiv/a2c_"
        if np.mean(reward_memory) > 3:
            model_path_template_sfx = "_{}_r{:.2f}.h5".format(episode, np.mean(reward_memory))
        else:
            model_path_template_sfx = "_latest.h5"
        actor.save(model_path_template_pfx + "actor" + model_path_template_sfx)
        critic.save(model_path_template_pfx + "critic" + model_path_template_sfx)

    episode += 1
