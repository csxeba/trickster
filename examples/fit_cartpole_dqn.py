import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling
from trickster.experience import Experience
from trickster.utility import visual, history

env = gym.make("CartPole-v1")
input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = Sequential([Dense(400, activation="relu", input_shape=input_shape),
                  Dense(300, activation="relu"),
                  Dense(num_actions, activation="linear")])
ann.compile(loss="mse", optimizer=Adam(1e-3))

agent = DQN(ann,
            action_space=env.action_space,
            memory=Experience(max_length=10000),
            epsilon=1.,
            epsilon_decay=0.99995,
            epsilon_min=0.1,
            discount_factor_gamma=0.98,
            use_target_network=False)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=300))
test_rollout = Trajectory(agent, gym.make("CartPole-v1"))

learning_history = history.History("reward_sum", *agent.history_keys, "epsilon")

for episode in range(1, 301):

    for update in range(32):
        rollout.roll(steps=2, verbose=0, push_experience=True)
        agent_history = agent.fit(batch_size=32, verbose=0)
        learning_history.buffer(**agent_history)

    test_history = test_rollout.rollout(verbose=0, push_experience=False)
    learning_history.push_buffer()
    learning_history.record(reward_sum=test_history["reward_sum"], epsilon=agent.epsilon)

    learning_history.print(average_last=10, return_carriege=True, prefix="Episode {:>4}".format(episode))

    if episode % 10 == 0:
        print()

visual.plot_history(learning_history, smoothing_window_size=100, skip_first=0)
