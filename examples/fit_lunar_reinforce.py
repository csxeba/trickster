import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import REINFORCE
from trickster.rollout import Trajectory
from trickster.experience import Experience
from trickster.utility import visual, history

env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(24, activation="tanh", input_shape=input_shape),
                     Dense(24, activation="tanh"),
                     Dense(num_actions, activation="softmax")])
policy.compile(loss="categorical_crossentropy", optimizer=Adam(1e-3))

agent = REINFORCE(policy,
                  action_space=num_actions,
                  memory=Experience(max_length=10000),
                  discount_factor_gamma=0.99)

rollout = Trajectory(agent, env)
hst = history.History("reward_sum", "loss", "entropy")

for episode in range(1, 1001):
    for roll in range(4):
        rollout_history = rollout.rollout(verbose=0, push_experience=True, render=False)
    agent_history = agent.fit(batch_size=-1, verbose=0, reset_memory=True)
    hst.record(reward_sum=rollout_history["reward_sum"], **agent_history)
    hst.print(average_last=100, templates={
        "reward_sum": "{:.4f}", "loss": "{:.4f}", "entropy": "{:.4f}", "epsilon": "{:.2%}"
    }, return_carriege=True, prefix="Episode {:>4}".format(episode))
    if episode % 100 == 0:
        print()

visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
