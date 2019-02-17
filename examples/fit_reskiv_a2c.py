from collections import deque

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Input, LeakyReLU, Activation
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import SGD

from trickster.agent import A2C
from trickster.rollout import MultiRolling, Trajectory, RolloutConfig
from trickster.experience import Experience

from grund.reskiv import ReskivConfig, Reskiv
from grund.util.movement import get_movement_vectors

TRAIN = True
WARMUP = 1
NUM_MOVES = 4
NUM_PARALLEL_ROLLOUTS = 4

MOVES = get_movement_vectors(num_directions=NUM_MOVES)

rcfg = ReskivConfig(canvas_shape=[64, 64, 3], player_radius=3, target_size=3)
envs = [Reskiv(rcfg) for _ in range(NUM_PARALLEL_ROLLOUTS)]
test_env = Reskiv(rcfg)

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
actor.compile(SGD(1e-4, momentum=0.9), "categorical_crossentropy")
critic = Model(critic_input, value_estimate, name="Critic")
critic.compile(SGD(5e-4, momentum=0.9), "mse")

agent = A2C(actor, critic,
            action_space=MOVES,
            memory=Experience(max_length=10000),
            discount_factor_gamma=0.995,
            entropy_penalty_coef=0.0,
            state_preprocessor=lambda state: state / 255.)

episode = 1

reward_memory = deque(maxlen=10)
step_lengths = deque(maxlen=10)
critic_losses = deque(maxlen=10)
actor_losses = deque(maxlen=10)
actor_utility = deque(maxlen=10)
actor_entropy = deque(maxlen=10)

rollout = MultiRolling(agent, envs, rollout_configs=RolloutConfig(max_steps=512, skipframes=2))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=512, skipframes=2))

while 1:
    episode_a_losses = []
    episode_a_utility = []
    episode_a_entropy = []
    episode_c_losses = []

    for update in range(32):
        rollout.roll(steps=2, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)

        episode_a_losses.append(agent_history["actor_loss"])
        episode_a_utility.append(agent_history["actor_utility"])
        episode_a_entropy.append(agent_history["actor_entropy"])
        episode_c_losses.append(agent_history["critic_loss"])

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    reward_memory.append(test_history["reward_sum"])
    step_lengths.append(test_history["steps"])
    actor_losses.append(np.mean(episode_a_losses))
    actor_utility.append(np.mean(episode_a_utility))
    actor_entropy.append(np.mean(episode_a_entropy))
    critic_losses.append(np.mean(episode_c_losses))

    print("\rEPISODE {:>4} STEP: {:>7.2f} RRWD: {: >5.1f} ALOSS {: >7.4f} UTIL {: >7.4f} ENTR {:>6.4f} CRIT {:.4f}"
        .format(episode,
                np.mean(step_lengths),
                np.mean(reward_memory),
                np.mean(actor_losses),
                np.mean(actor_utility),
                np.mean(actor_entropy),
                np.mean(critic_losses)), end="")

    if episode % 10 == 0:
        print()

    episode += 1
