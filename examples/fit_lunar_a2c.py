import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import A2C
from trickster.rollout import Rolling, Trajectory, RolloutConfig
from trickster.utility import visual, history

env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = Sequential([Dense(24, activation="relu", input_shape=input_shape),
                    Dense(24, activation="relu"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-4))

critic = Sequential([Dense(24, activation="relu", input_shape=input_shape),
                    Dense(24, activation="relu"),
                    Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(5e-4))

agent = A2C(actor,
            critic,
            action_space=num_actions,
            absolute_memory_limit=10000,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0)

rollout = Rolling(agent.create_workers(1)[0], gym.make("LunarLander-v2"))
test_rollout = Trajectory(agent, gym.make("LunarLander-v2"))

hst = history.History("reward_sum", "actor_loss", "actor_utility", "actor_utility_std", "actor_entropy", "critic_loss")

agent_history = {}

for episode in range(1, 1001):

    rollout.roll(steps=8, verbose=0, push_experience=True)
    agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)

    roll_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)

    hst.record(reward_sum=roll_history["reward_sum"], **agent_history)
    hst.print(templates={"reward_sum": "{:>3.0f}",
                         "actor_loss": "{:.4f}",
                         "actor_utility": "{:.4f}",
                         "actor_utility_std": "{:.4f}",
                         "actor_entropy": "{:.4f}",
                         "critic_loss": "{:.4f}"},
              return_carriege=True, prefix="Episode {:>5}".format(episode), average_last=100)

    if episode % 100 == 0:
        print()

visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
