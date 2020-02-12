import numpy as np
import gym

from trickster.agent import DQN
from trickster.rollout import Trajectory, RolloutConfig, Rolling
from trickster.model import mlp
from trickster.experience import replay_buffer

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

ann = mlp.wide_mlp_critic(input_shape, num_actions, adam_lr=1e-3)

agent = DQN(ann,
            action_space=env.action_space,
            training_memory=replay_buffer.Experience(max_length=1000),
            epsilon=1.,
            epsilon_decay=0.9995,
            epsilon_min=0.1,
            discount_gamma=0.99,
            use_target_network=False)

rollout = Rolling(agent, env, config=RolloutConfig(max_steps=200))
test_rollout = Trajectory(agent, test_env, config=RolloutConfig(max_steps=200))

rollout.roll(steps=1000, verbose=0, learning=True)

for epoch in range(1, 101):
    losses = []
    qs = []
    for update in range(1, 301):
        data = agent.fit(batch_size=32)
        losses.append(data["loss"])
        qs.append(data["Q"])
    print("{} - loss {:.4f} - q {:.4f}".format(epoch, np.mean(losses), np.mean(qs)))
