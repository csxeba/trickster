
[![codebeat badge](https://codebeat.co/badges/f72db301-fd66-4c05-b1ca-9b8c8196f06e)](https://codebeat.co/projects/github-com-csxeba-trickster-master)

# Trickster

Deep Reinforcement Learning mini-library with the aim of clear implementation of some algorithms.

The supported Deep Learning engine is **TensorFlow > 2.0**. An older version of the library, which supported
the Multi-Backend Keras is available through the mb-keras branch.

## Installation

To install trickster, simply run

`pip3 install git+https://github.com/csxeba/trickster.git`

## Documentation

The *trickster* library is organized around three basic concepts:

**Environment**: a game or some playground, where an entity can be placed into, interacting with the environment in the form of providing
actions and receiving states and rewards.

**Agent**: the entity which acts in the environment.

**Rollout**: orchestrates the interactions between the *Agent* and the *Environment*.

### Environment

A game or other playground, where a player or entity can be placed into, interacting with the environment in the form of providing
actions and receiving states and rewards.

The environments used with *Trickster* must present an OpenAI Gym-like interface.

### Agent

Available through *trickster.agent*

The agent is a wrapper around one or more **tf.keras Models**. This handles the learning and experience collection from the environment.
Agents are written so that they are maximizing the expected reward of the environment they are interacting with.

All agent classes present the .from_environment() factory method, which inspects the action and observation spaces of
an environment and automatically selects (relatively small) Neural Networks for running on the given environment.

These hardcoded architectures can be overriden by passing an architecture to the factory method 

### Rollouts

Available in *trickster.rollout*.

Rollout is the concept of combining an agent with an environment.
There are two types of rollouts in *Trickster*:
- **Trajectory**: a complete trajectory from start to the 'done' flag. It can be used for testing an agent
or for *Monte Carlo* learning.
- **Rolling**: this type of rollout is for *Time Difference* learning. A fixed number of steps
are executed in the environment. The environment is reset whenever a *done* flag is received.

Both **Trajectory** and **Rolling** are available in a multi-environment configuration for parallel execution
of environment instances. These classes are called:
- **MultiTrajectory**: Trivially parallelizable, yet I didn't have time to parallelize it as of today...
- **MultiRolling**: Roll the agent in several environments.

*Rollout* types expect the following constructor arguments:

- **agent**: an object of one of the *Agent* subclasses.
- **env**: in non-multi classes. An object, presenting the *Gym Environment* interface
- **envs**: in multi classes. A list of environments, which can't have the same object ID.
- **config**: in non-multi classes. An instance of *RolloutConfig*. Optional, see defaults below.
- **rollout_configs**: in multi classes. Either an instance of *RolloutConfig* or one for every env passed.

*Trajectory* type rollouts present the following public methods:
- **rollout(verbose, push_experience, render)**: sample a complete trajectory. Optionally save the experience or render.

*Rolling* type rollouts present the following public methods:
- **roll(steps, verbose, push_experience)**: execute the environment/agent for a given number of timesteps.

Both rollout types present a *fit()* method, which orchestrates basic learning
functionality. See the docstrings for documentation and argument list.

## Working Examples

Working examples are available in the repo under the *examples* folder.
Not all algorithms converge on all environments. Please note that hyperparameters are not optimized for these runs.
