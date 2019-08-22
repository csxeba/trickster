import gym
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import mlflow

from trickster.agent import DDPG
from trickster.rollout import Trajectory, Rolling, RolloutConfig
from trickster.utility import visual, spaces, history
from trickster.experience import Experience


def build_actor():
    actor = Sequential([Dense(400, activation="relu", input_shape=input_shape),
                        Dense(300, activation="relu"),
                        Dense(num_actions, activation="tanh")])
    actor.compile(loss="mse", optimizer=Adam(1e-4))

    return actor


def build_critic():
    critic_state_input = Input(input_shape)
    critic_action_input = Input([num_actions])

    x = Dense(400, activation="relu", kernel_regularizer=l2(0.01))(critic_state_input)
    x = concatenate([x, critic_action_input], axis=1)
    x = Dense(300, activation="relu", kernel_regularizer=l2(0.01))(x)
    q = Dense(1, activation="linear")(x)

    critic = Model([critic_state_input, critic_action_input], q)
    critic.compile(optimizer=Adam(1e-3), loss="mse")

    return critic


env = gym.make("LunarLanderContinuous-v2")

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

actor, critic = build_actor(), build_critic()

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(max_length=int(1e4)),
             discount_factor_gamma=0.99,
             action_noise_sigma=1.,
             action_noise_sigma_decay=1.,
             min_action_noise_sigma=0.1,
             action_minima=-1,
             action_maxima=1)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env, RolloutConfig(testing_rollout=True))

mlflow.set_tracking_uri("../artifacts/TestRuns")
learning_history = history.History("reward_sum", "actor_loss", "critic_loss", "noise_sigma")

rollout.roll(2048, verbose=0, push_experience=True)

for episode in range(1, 1001):

    for update in range(1, 129):
        rollout.roll(steps=1, verbose=0, push_experience=True)
        fit_actor = update % 2 == 0
        agent_history = agent.fit(updates=1, batch_size=64, fit_actor=fit_actor, update_target_tau=0.01)
        learning_history.buffer(**agent_history)

    roll_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
    learning_history.push_buffer()
    learning_history.record(reward_sum=roll_history["reward_sum"], noise_sigma=agent.action_noise_sigma)
    learning_history.print(average_last=100,
                           templates={"reward_sum": "{:> 8.2f}", "actor_loss": "{:> 7.4f}", "critic_loss": "{:>7.4f}"},
                           prefix="Episode: {:>4}".format(episode),
                           return_carriege=True)
    for k, v in learning_history.last().items():
        mlflow.log_metric(k, v, step=episode)

visual.plot_history(learning_history, smoothing_window_size=100)
