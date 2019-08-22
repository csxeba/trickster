import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import DQN
from trickster.rollout import Rolling, Trajectory
from trickster.experience import Experience
from trickster.utility import visual, history

env = gym.make("LunarLander-v2")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

policy = Sequential([Dense(24, activation="relu", input_shape=input_shape),
                     Dense(24, activation="relu"),
                     Dense(num_actions, activation="linear")])
policy.compile(loss="mse", optimizer=Adam(1e-4))

agent = DQN(policy,
            action_space=num_actions,
            memory=Experience(max_length=10000),
            epsilon=1.,
            epsilon_decay=1.,
            epsilon_min=0.1,
            discount_factor_gamma=0.99,
            use_target_network=True)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env)

hst = history.History("reward_sum", "loss", "epsilon")

for warmup in range(1, 33):
    rollout.roll(32, verbose=0, push_experience=True)
agent.epsilon_decay = 0.99999

for episode in range(1, 1001):
    rollout.roll(steps=32, verbose=0, push_experience=True)
    agent_history = agent.fit(updates=10, batch_size=64, verbose=0)
    test_history = test_rollout.rollout(verbose=0, push_experience=False, render=False)
    hst.record(reward_sum=test_history["reward_sum"], loss=agent_history["loss"], epsilon=agent.epsilon)
    hst.print(average_last=100, templates={
        "reward_sum": "{:.4f}", "loss": "{:.4f}", "epsilon": "{:.2%}"
    }, return_carriege=True, prefix="Episode {:>4}".format(episode))
    agent.meld_weights(mix_in_ratio=0.01)
    if episode % 100 == 0:
        print()

visual.plot_history(hst, smoothing_window_size=100, skip_first=10)
