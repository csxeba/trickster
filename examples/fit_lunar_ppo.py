import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import mlflow

from trickster.agent import PPO
from trickster.rollout import Trajectory
from trickster.utility import visual, history

env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

actor = Sequential([Dense(24, activation="tanh", input_shape=input_shape),
                    Dense(24, activation="tanh"),
                    Dense(num_actions, activation="softmax")])
actor.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3))

critic = Sequential([Dense(24, activation="tanh", input_shape=input_shape),
                     Dense(24, activation="tanh"),
                     Dense(1, activation="linear")])
critic.compile(loss="mse", optimizer=Adam(1e-3))

agent = PPO(actor,
            critic,
            action_space=num_actions,
            discount_factor_gamma=0.99,
            entropy_penalty_coef=0.05)

rollout = Trajectory(agent.create_workers(1)[0], env)

mlflow.set_tracking_uri("../artifacts/TestRuns")
hst = history.History("reward_sum", *agent.history_keys)

for episode in range(1, 1001):

    for roll in range(16):
        rollout_history = rollout.rollout(verbose=0, push_experience=True, render=False)
        hst.buffer(reward_sum=rollout_history["reward_sum"])

    agent_history = agent.fit(epochs=-1, batch_size=32, verbose=0, reset_memory=True)
    agent_history["actor_kld"] *= 1000

    hst.push_buffer()
    hst.record(**agent_history)
    for key in agent.history_keys:
        mlflow.log_metric(key, agent_history[key])
    hst.print(average_last=100, templates={
        k: "{:.4f}" for k in ["reward_sum"] + list(agent.history_keys)
    })
    if episode % 100 == 0:
        print()

mlflow.end_run()
visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
