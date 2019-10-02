from tensorflow import keras

from trickster.agent import DQN
from trickster.rollout import Rolling, Trajectory
from trickster.utility import gymic
from trickster.model import mlp


def least_bellman_loss(y_true, y_pred):
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    l2_penalty = keras.backend.mean(keras.backend.square(y_pred), axis=-1)
    return mse + L2_LAMBDA * l2_penalty


L2_LAMBDA = 0.01

env = gymic.rwd_scaled_env("LunarLander-v2")
test_env = gymic.rwd_scaled_env("LunarLander-v2")

input_shape = env.observation_space.shape
num_actions = env.action_space.n

model = mlp.wide_mlp_critic_network(input_shape, num_actions, adam_lr=1e-3)
model.compile(keras.optimizers.Adam(1e-4), least_bellman_loss)

agent = DQN(model,
            action_space=env.action_space,
            epsilon=1.,
            epsilon_decay=1.,
            epsilon_min=0.1,
            discount_factor_gamma=0.99,
            use_target_network=True)

rollout = Rolling(agent, env)
test_rollout = Trajectory(agent, env)

rollout.roll(10_000, verbose=0, push_experience=True)  # warmup
agent.epsilon_decay = 0.99999

rollout.fit(episodes=1000, updates_per_episode=64, step_per_update=1, update_batch_size=32,
            testing_rollout=test_rollout, plot_curves=True, render_every=100)
test_rollout.render(repeats=100)
