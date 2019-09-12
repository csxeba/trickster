import gym


def gym_env(env_id, observation_callback=None, reward_callback=None, action_callback=None):

    wrappers = []

    if observation_callback is not None:
        class ObsW(gym.ObservationWrapper):
            def observation(self, observation):
                return observation_callback(observation)
        wrappers.append(ObsW)

    if reward_callback is not None:
        class RwdW(gym.RewardWrapper):
            def reward(self, reward):
                return reward_callback(reward)
        wrappers.append(RwdW)

    if action_callback is not None:
        class ActW(gym.ActionWrapper):
            def action(self, action):
                return action_callback(action)
        wrappers.append(ActW)

    env = gym.make(env_id)
    for w in wrappers:
        env = w(env)

    return env


def rwd_scaled_cartpole(reward_scale=0.01):
    return gym_env("CartPole-v1", reward_callback=lambda r: r * reward_scale)
