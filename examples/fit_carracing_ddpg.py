import keras
import gym

from trickster.agent import DDPG
from trickster.rollout import Trajectory, MultiRolling, RolloutConfig
from trickster.experience import Experience
from trickster.utility import spaces


K = keras.backend


def _activate(_x, activation="leakyrelu", batch_normalize=True):
    if batch_normalize:
        _x = keras.layers.BatchNormalization()(_x)
    if activation == "leakyrelu":
        _x = keras.layers.LeakyReLU()(_x)
    else:
        _x = keras.layers.Activation(activation)(_x)
    return _x


def _conv(_x, width, stride=1, activation="leakyrelu", batch_normalize=True):
    _x = keras.layers.Conv2D(width, kernel_size=3, strides=stride, padding="same")(_x)
    _x = _activate(_x, activation, batch_normalize)
    return _x


def _dense(_x, width, activation="leakyrelu", batch_normalize=True):
    _x = keras.layers.Dense(width)(_x)
    _x = _activate(_x, activation, batch_normalize)
    return _x


def _clip(_x, low, high):
    _x = keras.layers.Lambda(lambda _xx: K.clip(_xx, low, high))(_x)
    return _x


def make_backbone():
    state_in = keras.Input(input_shape, name="state_in")
    x = _conv(state_in, width=8, stride=2, batch_normalize=BATCH_NORMALIZE)  # 48
    x = _conv(x, width=16, stride=2, batch_normalize=BATCH_NORMALIZE)  # 24
    x = _conv(x, width=32, stride=2, batch_normalize=BATCH_NORMALIZE)  # 12
    features = keras.layers.GlobalAveragePooling2D()(x)  # 32
    return state_in, features


def make_actor(state_in, features):

    actor_stream = _dense(features, width=64, batch_normalize=BATCH_NORMALIZE)
    actor_stream = _dense(actor_stream, width=64, batch_normalize=BATCH_NORMALIZE)

    actor_output0 = _clip(_dense(actor_stream, width=1, batch_normalize=False), -1, 1)
    actor_output1 = _clip(_dense(actor_stream, width=1, batch_normalize=False), 0, 1)
    actor_output2 = _clip(_dense(actor_stream, width=1, batch_normalize=False), 0, 1)

    actor_output = keras.layers.concatenate([actor_output0, actor_output1, actor_output2])

    actor_network = keras.Model(state_in, actor_output)
    actor_network.compile(keras.optimizers.Adam(ACTOR_LR), keras.losses.mean_squared_error)

    return actor_network


def make_critic(state_in, action_in, features):

    critic_stream = keras.layers.concatenate([features, action_in])
    critic_stream = _dense(critic_stream, width=64, batch_normalize=BATCH_NORMALIZE)
    critic_stream = _dense(critic_stream, width=64, batch_normalize=BATCH_NORMALIZE)

    critic_output = _dense(critic_stream, width=1, batch_normalize=False)

    critic_network = keras.Model([state_in, action_in], critic_output)
    critic_network.compile(keras.optimizers.Adam(CRITIC_LR), keras.losses.mean_squared_error)

    return critic_network


class CarRacing(gym.ObservationWrapper):

    def __init__(self):
        super().__init__(env=gym.make("CarRacing-v0"))

    def observation(self, observation):
        return observation / 255.


BATCH_NORMALIZE = False
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
NUM_ENVS = 4

envs = [CarRacing() for _ in range(NUM_ENVS)]
test_env = CarRacing()

input_shape = envs[0].observation_space.shape  # 96 96 3
num_actions = envs[0].action_space.shape[0]

state_inputs, backbone_features = make_backbone()

actor = make_actor(state_inputs, backbone_features)

action_inputs = keras.Input([num_actions], name="critic_action_in")
critic = make_critic(state_inputs, action_inputs, backbone_features)

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(max_length=int(1e4)),
             discount_factor_gamma=1.,
             action_noise_sigma=0.1,
             action_noise_sigma_decay=1.,
             action_minima=[-1, 0, 0],
             action_maxima=[1, 1, 1])

rollout = MultiRolling(agent, envs)
test_rollout = Trajectory(agent, test_env, RolloutConfig(max_steps=128))

rollout.fit(episodes=10000, updates_per_episode=4z6vgbhn    , steps_per_update=4, update_batch_size=64,
            testing_rollout=test_rollout, render_every=100)

test_rollout.render(repeats=100)
