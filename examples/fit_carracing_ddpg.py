import keras
import gym

from trickster.agent import DDPG
from trickster.rollout import Trajectory, Rolling, RolloutConfig
from trickster.experience import Experience
from trickster.model import cnn


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


def _dense(_x, width, activation="leakyrelu", batch_normalize=True):
    _x = keras.layers.Dense(width)(_x)
    if batch_normalize:
        _x = keras.layers.BatchNormalization()(_x)
    if activation == "leakyrelu":
        _x = keras.layers.LeakyReLU()(_x)
    else:
        _x = keras.layers.Activation(activation)(_x)
    return _x


env = gym.make("CarRacing-v0")

input_shape = env.observation_space.shape  # 96 96 3
num_actions = env.action_space.shape[0]

state_in = keras.Input(input_shape)
action_in = keras.Input([num_actions])

x = _conv(state_in, width=8, stride=2)  # 48
x = _conv(x, width=16, stride=2)  # 24
x = _conv(x, width=32, stride=2)  # 12
features = keras.layers.GlobalAveragePooling2D()(x)  # 32

actor_stream = keras.layers.Dense()

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(max_length=int(1e4)),
             discount_factor_gamma=1.,
             action_noise_sigma=0.1,
             action_noise_sigma_decay=0.99,
             action_minima=-2,
             action_maxima=2)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env, RolloutConfig(testing_rollout=True))

rollout.fit(episodes=10000, updates_per_episode=64, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout)
test_rollout.render(repeats=10)
