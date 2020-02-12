import gym


def wrap(env, reward_callback=None, action_callback=None, state_callback=None):

    callbacks = []

    if reward_callback is not None:

        class RewardCB(gym.RewardWrapper):

            def reward(self, reward):
                return reward_callback(reward)

        callbacks.append(RewardCB)

    if action_callback is not None:

        class ActionCB(gym.ActionWrapper):

            def action(self, action):
                return action_callback(action)

        callbacks.append(ActionCB)

    if state_callback is not None:

        class StateCB(gym.ObservationWrapper):

            def observation(self, observation):
                return state_callback(observation)

        callbacks.append(StateCB)

    for callback in callbacks:
        env = callback(env)

    return env
