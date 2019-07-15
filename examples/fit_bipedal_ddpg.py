import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam

from trickster.agent import DDPG
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.utility import visual, spaces, history
from trickster.experience import Experience


class PendulumEnv(gym.RewardWrapper):

    def __init__(self):
        super().__init__(gym.make("Pendulum-v0"))

    def reward(self, reward):
        return reward / 16.2736044


def build_actor():
    actor = Sequential([Dense(400, activation="relu", input_shape=input_shape),
                        Dense(300, activation="relu"),
                        Dense(num_actions, activation="tanh")])
    actor.compile(loss="mse", optimizer=Adam(1e-4))

    return actor


def build_critic():
    critic_state_input = Input(input_shape)
    critic_action_input = Input([num_actions])

    x = Dense(300, activation="relu")(critic_state_input)
    x = concatenate([x, critic_action_input], axis=1)
    x = Dense(400, activation="relu")(x)
    q = Dense(1, activation="linear")(x)

    critic = Model([critic_state_input, critic_action_input], q)
    critic.compile(optimizer=Adam(1e-3), loss="mse")

    return critic


env = PendulumEnv()

input_shape = env.observation_space.shape
num_actions = env.action_space.shape[0]


actor, critic = build_actor(), build_critic()

agent = DDPG(actor, critic,
             action_space=spaces.CONTINUOUS,
             memory=Experience(),
             discount_factor_gamma=0.99)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, PendulumEnv())

learning_history = history.History("reward_sum", "actor_loss", "critic_loss")

for episode in range(1, 5001):

    rollout.roll(steps=2, verbose=0, push_experience=True)
    agent_history = agent.fit(updates=1, batch_size=128)
    agent.meld_weights(1e-3, 1e-3)

    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    learning_history.record(**agent_history)
    learning_history.record(**test_history)

    learning_history.print(average_last=100,
                           templates={"reward_sum": "{:> 8.2f}", "actor_loss": "{:> 7.4f}", "critic_loss": "{:>7.4f}"},
                           prefix="Episode: {:>4}".format(episode),
                           return_carriege=True)
    if episode % 100 == 0:
        print()

visual.plot_history(learning_history, smoothing_window_size=10)
