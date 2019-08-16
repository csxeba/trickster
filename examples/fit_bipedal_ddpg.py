import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam

from trickster.agent import DDPG
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.utility import visual, spaces, history
from trickster.experience import Experience


def build_actor():
    actor = Sequential([Dense(400, activation="relu", input_shape=input_shape, kernel_regularizer="l2"),
                        Dense(300, activation="relu", kernel_regularizer="l2"),
                        Dense(num_actions)])
    actor.compile(loss="mse", optimizer=Adam(1e-3))

    return actor


def build_critic():
    critic_state_input = Input(input_shape)
    critic_action_input = Input([num_actions])

    x = Dense(300, activation="relu", kernel_regularizer="l2")(critic_state_input)
    x = concatenate([x, critic_action_input], axis=1)
    x = Dense(400, activation="relu", kernel_regularizer="l2")(x)
    q = Dense(1, activation="linear")(x)

    critic = Model([critic_state_input, critic_action_input], q)
    critic.compile(optimizer=Adam(1e-4), loss="mse")

    return critic


env = gym.make("Pendulum-v0")

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]

actor, critic = build_actor(), build_critic()

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(max_length=int(1e6)),
             discount_factor_gamma=0.98,
             action_noise_sigma=3.,
             action_noise_sigma_decay=1.,
             min_action_noise_sigma=0.1)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300, testing_rollout=False))
test_rollout = Trajectory(agent, gym.make("Pendulum-v0"), config=RolloutConfig(max_steps=None, testing_rollout=True))

learning_history = history.History("reward_sum", "actor_loss", "critic_loss", "noise_sigma")

fit_actor = False
for episode in range(1, 5001):

    rollout.roll(steps=128, verbose=0, push_experience=True)
    agent_history = agent.fit(updates=4, batch_size=32, fit_actor=fit_actor)
    agent.meld_weights(1e-2, 1e-2)

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    learning_history.record(**agent_history)
    learning_history.record(**test_history)
    learning_history.record(noise_sigma=agent.action_noise_sigma)

    learning_history.print(average_last=100,
                           templates={"reward_sum": "{:> 8.2f}", "actor_loss": "{:> 7.4f}", "critic_loss": "{:>7.4f}",
                                      "noise_sigma": "{:.4f}"},
                           prefix="Episode: {:>4}".format(episode),
                           return_carriege=True)
    if episode % 100 == 0:
        print()
        fit_actor = True
        agent.action_noise_sigma_decay = 0.99999

visual.plot_history(learning_history, smoothing_window_size=10)
