import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import PPO
from trickster.rollout import Rolling, Trajectory
from trickster.utility import visual, history


class Lunar(gym.RewardWrapper):

    def __init__(self):
        super().__init__(gym.make("LunarLander-v2"))

    def reward(self, reward):
        return reward / 100.


env = Lunar()
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = Sequential([Dense(400, activation="relu", input_shape=input_shape),
                    Dense(300, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(400, activation="relu", input_shape=input_shape),
                     Dense(300, activation="relu"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(1e-4))

agent = PPO(actor,
            critic,
            action_space=num_actions,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.05)

rollout = Rolling(agent.create_workers(1)[0], env)
test_rollout = Trajectory(agent.create_workers(1)[0], Lunar())

hst = history.History("reward_sum", *agent.history_keys)

for episode in range(1, 1001):

    for roll in range(4):
        rollout.roll(steps=128, verbose=0, push_experience=True)
        agent_hst = agent.fit(epochs=3, batch_size=32, verbose=0, fit_actor=True, fit_critic=True, reset_memory=True)
        agent_hst["kld"] *= 1000
        hst.buffer(**agent_hst)

    for test_roll in range(4):
        rollout_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
        hst.buffer(**rollout_history)

    hst.push_buffer()
    hst.print(average_last=100, return_carriege=True, prefix="Episode {:>5}".format(episode))
    if episode % 100 == 0:
        print()

visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
