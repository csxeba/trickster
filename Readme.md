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

## On environments

Environments used must present an OpenAI Gym-like interface, or at least a step(action) method returning (state, reward, done, info).

Windozers: check out Grund, my other homebrew-library for simple RL environmnets compatible with this lib.
Grund only depends on NumPy and OpenCV, so installation shouldn't be as problematic as it might be with Gym.
