# Trickster

Deep reinforcement learning homebrew-library

Currently Keras is supported and tested as a neural network engine.

## Installation

To install trickster, simply run
`pip3 install git+https://github.com/csxeba/trickster.git`

## Algorithms

- REINFORCE
- DQN
- A2C

There is also a class named Policy, which is a light wrapper without learning logic to test and inspect trained policy agents in an environment.

## On environments

Environments used must present an OpenAI Gym-like interface, or at least a step(action) method returning (state, reward, done, info).

Windozers: check out Grund, my other homebrew-library for simple RL environmnets compatible with this lib.
Grund only depends on NumPy and OpenCV, so installation shouldn't be as problematic as it might be with Gym.

## The library

Central concepts of the library are as follows:

### Environment
A game or other playground, where a player or entity can be placed into, interacting with the environment in the form of providing
actions and receiving states and rewards.

### Agent
The agent is a wrapper around a neural network, which handles the learning and experience collection from the environment. Agents are written
so that they are MAXIMIZING the expected reward of the environment they are interacting with.

### Rollout
Love child of an environment-agent pair, which handles different run styles:
- Complete rollout: Monte Carlo style, where the agent only learns at the end of an episode.
- Single roll: for a fixed number of steps for time-difference learning (agents bootstrapping themselves)

Currently two Rollouts are implemented:
- Rollout is a single agent in a single environment
- MultiRollout is a single agent in multiple rollouts in parallel (not parallelized yet though)
