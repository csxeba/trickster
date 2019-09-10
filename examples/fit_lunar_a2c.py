import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import A2C
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

agent = A2C(actor,
            critic,
            action_space=num_actions,
            absolute_memory_limit=10000,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.05)

rollout = Rolling(agent.create_workers(1)[0], env)
test_rollout = Trajectory(agent, Lunar())


def train():
    hst = history.History("reward_sum", "actor_loss", "actor_utility", "actor_utility_std", "actor_entropy",
                          "values", "advantages", "critic_loss")


    for episode in range(1, 1001):

        for update in range(1, 4):
            rollout.roll(steps=128, verbose=0, push_experience=True)
            agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)
            hst.buffer(**agent_history)

        for test_round in range(1, 4):
            roll_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
            hst.buffer(reward_sum=roll_history["reward_sum"])

        hst.push_buffer()
        hst.print(return_carriege=True, prefix="Episode {:>5}".format(episode), average_last=100)

        if episode % 100 == 0:
            print()
            actor.save_weights("actor_episode_{}.h5".format(episode))
            critic.save_weights("critic_episode_{}.h5".format(episode))

    visual.plot_history(hst, smoothing_window_size=100, skip_first=10)


def evaluate(episode=1000):
    actor.load_weights("actor_episode_{}.h5".format(episode))
    while 1:
        test_rollout.rollout(verbose=1, push_experience=False, render=True)
        print()


if __name__ == '__main__':
    evaluate()
