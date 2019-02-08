from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, LeakyReLU, Activation
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

actor_input = Input(shape=[64, 64, 3], name="actor_input")
critic_input = Input(shape=[64, 64, 3], name="critic_input")

actor_stream = Conv2D(16, (4, 4), strides=(2, 2), padding="same")(actor_input)  # 32
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(32, (4, 4), strides=(2, 2), padding="same")(actor_stream)  # 16
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(actor_stream)  # 8
actor_stream = LeakyReLU()(actor_stream)
actor_stream = Conv2D(NUM_MOVES, (1, 1))(actor_stream)
actor_stream = GlobalAveragePooling2D()(actor_stream)
# actor_stream = Flatten()(actor_stream)
action_probs = Activation("softmax")(actor_stream)

critic_stream = Conv2D(16, (4, 4), strides=(2, 2), padding="same")(critic_input)  # 32
critic_stream = LeakyReLU()(critic_stream)
critic_stream = Conv2D(32, (4, 4), strides=(2, 2), padding="same")(critic_stream)  # 16
critic_stream = LeakyReLU()(critic_stream)
critic_stream = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(critic_stream)  # 8
critic_stream = LeakyReLU()(critic_stream)
critic_stream = Conv2D(1, (1, 1), padding="valid")(critic_stream)
value_estimate = GlobalAveragePooling2D()(critic_stream)
# value_estimate = Flatten()(critic_stream)

actor = Model(actor_input, action_probs, name="Actor")
actor.compile(Adam(1e-4), "categorical_crossentropy")
critic = Model(critic_input, value_estimate, name="Critic")
critic.compile(Adam(5e-4), "mse")

agent = A2C(actor, critic, actions=MOVES, memory=Experience(max_length=10000), reward_discount_factor=0.99,
            state_preprocessor=lambda state: state / 255.)

screen = CV2Screen(scale=2)
episode = 1
reward_memory = deque(maxlen=10)
critic_losses = deque(maxlen=10)
actor_losses = deque(maxlen=10)

rollout = MultiRollout(agent, envs, warmup_episodes=WARMUP, rollout_configs=RolloutConfig(max_steps=512))

while 1:
    rollout.reset()
    episode_actor_losses = []
    episode_entropy = []
    episode_critic_losses = []
    while 1:
        rollout_history = rollout.roll(steps=2, verbose=0, learning_batch_size=0)
        if rollout.finished:
            break
        agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)
        reward_memory.append(rollout_history["reward_sum"])
        episode_actor_losses.append(agent_history["actor_utility"])
        episode_critic_losses.append(agent_history["critic_loss"])

    actor_losses.append(np.mean(episode_actor_losses))
    critic_losses.append(np.mean(episode_critic_losses))
    print("\rEPISODE {} RRWD: {:.2f} ACTR {:.4f} CRIT {:.4f}".format(
        episode,
        np.mean(reward_memory),
        np.mean(actor_losses),
        np.mean(critic_losses)), end="")

    if episode % 10 == 0:
        print()

    episode += 1
