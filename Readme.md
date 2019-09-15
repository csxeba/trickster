
[![codebeat badge](https://codebeat.co/badges/f72db301-fd66-4c05-b1ca-9b8c8196f06e)](https://codebeat.co/projects/github-com-csxeba-trickster-master)

# Trickster

Deep Reinforcement Learning mini-library with the aim of clear implementation of some algorithms.

Currently Keras is supported and tested as a deep learning engine. The library is not yet parallelized.

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

The agent is a wrapper around a Keras Model, which handles the learning and experience collection from the environment. Agents are written
so that they are maximizing the expected reward of the environment they are interacting with.

*Agents* have the following constructor parameters in common:

- **action_space**: action space, integer or an iterable holding possible actions to be taken
- **memory**: optional, an instance of Experience which is used as a buffer for learning
- **discount_factor_gamma**: I like long variable names
- **state_preprocessor**: callable, not quite stable yet. It is called on singe states and batches as well
 (this will change)
 
*Agents* present the following public methods:

- **model** or **actor** and **critic**: Keras Model instances
- **sample(state, reward, done)**: sample an action to be taken, given state. Also rewards and done flags.
- **push_experience(state, reward, done)**: direct experience is saved in an internal buffer. This method pushes
 it into the Experience buffer. Also handles the last reward and last done flag and for instance computes GAE.
- **fit(\*args, \*\*kwargs)**: updates the network parameters and optionally resets the memory buffer.
 returns a history dictionary holding losses. Specific algorithms have their own argument lists
 for *fit*.

### Exeprience

Generic *NumPy* ndarray-based buffer to store trajectories.

Constructor parameters:
- **max_length**

Public properties:
- **N**: number of samples currently in the buffer

Public methods:
- **reset()**: empties all arrays
- **remember(states, \*args)**: stores any number of arrays. The number only has to be consistent with the
number of arrays in the first call.
- **sample(size)**: samples a given number of trajectories. Returs (state, state_next, *)
- **stream(size, infinite)**: streams batches of <size>. Optionally streams infinitelly.

### Rollouts

Available in *trickster.rollout*.

Rollout is the concept of combining an agent with an environment.
There are two types of rollouts in *Trickster*:
- **Trajectory**: a complete trajectory from start to the 'done' flag. It can be used for testing an agent
or for *Monte Carlo* learning.
- **Rolling**: this type of rollout is for ie. *Time Difference* and bootstrap learning. A fixed number of steps
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
Not all algorithms converge on all environments. This might be due to
incorrect implementation, or incorrect hyperparameters or both...

One major takeaway for me regarding Deep Reinforcement Learning is the fact
of how unstable and unreliable it is.
