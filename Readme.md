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
- **reward_discount_factor_gamma**: I like long variable names
- **state_preprocessor**: callable, not quite stable yet. It is called on singe states and batches as well
 (this will change)
 
*Agents* present the following public methods:

- **model** or **actor** and **critic**: Keras Model instances
- **sample(state, reward, done)**: sample an action to be taken, given state. Also rewards and done flags.
- **push_experience(state, reward, done)**: direct experience is saved in an internal buffer. This method pushes
 it into the Experience buffer. Also handles the last reward and last done flag and for instance computes GAE.
- **fit(batch_size, verbose, reset_memory)**: updates the network parameters and optionally resets the memory buffer.
 returns a history dictionary holding losses.
- **fit(epochs, batch_size, verbose, reset_memory)**: PPO's interface for a multi-epoch update.

*DQN* and *DoubleDQN* specific constructor parameters:

- **epsilon**: Epsilon-greedy: probability of taking a uniform random action istead of arg max Q
- **epsilon_decay**: decays epsilon by this rate at every *sample()* call
- **epsilon_min**: minimum value of epsilon
- **use_target_network**: whether to use a target network for Bellman-target determination

*DQN* and *DoubleDQN* specific methods:

- **push_weights**: copy weights to target network
- **meld_weights(mix_in_ratio)**: target_network_weights = mix_in_ratio * new_weights + (1. - mix_in_ratio) * old_weights

*A2C* specific constructor parameters:
- **entropy_penalty_coef**: penalizes the negated entropy to increase exploration rate

*PPO* specific constructor parameters:
- **gae_factor_lambda**: coefficient for *Generalized Advantage Estimation*
- **entropy_penalty_coef**: penalizes the negated entropy to increase exploration rate
- **ratio_clip_epsilon**: clipping value for the probability ratio in the *PPO* clip surrogate loss function

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
- **rollout(verbose, push_experience)**: sample a complete trajectory. Optionally save the experience.

*Rolling* type rollouts present the following public methods:
- **roll(steps, verbose, push_experience)**: execute the environment/agent for a given number of timesteps.

## Working Examples

Working examples are available in the repo under the *examples* folder.

*CartPole* examples are checked for convergence, *Atari* examples aren't due to lack of time and compute :)
