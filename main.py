import os
import argparse

algo_help = "Algorithm to run, either on of REINFORCE, A2C, PPO, DQN, DoubleDQN, DDPG, TD3, SAC"

parser = argparse.ArgumentParser(
    prog="python main.py",
    description="Trickster - a Deep Reinforcement Learning library")
parser.add_argument("--env", "-e", help="Environment name from Gym, eg. CartPole-v1", required=True)
parser.add_argument("--algo", "-a", help=algo_help, required=True)
parser.add_argument("--max-steps", "-s", type=int, default=300,
                    help="Maximum number of steps in a rollout. Default: 300")
parser.add_argument("--parallel-envs", type=int, default=1,
                    help="How many environments to run in parallel during training. Default: 1")
parser.add_argument("--train-epochs", type=int, default=500,
                    help="Number of epochs to train for. Default: 500")
parser.add_argument("--batch-size", type=int, default=32,
                    help="Training batch size, if applicable. Default: 32")
parser.add_argument("--render-frequency", type=int, default=100,
                    help="During training, interpreted as epoch frequency. Default: 100")
parser.add_argument("--gpu", help="Which GPU to use. Defaults to no GPU (-1).", type=int, default=-1)
parser.add_argument("--do-plot", help="Whether to plot learning curves.", type=bool, default=False)
parser.add_argument("--render-final", help="Run the agent with rendering at the end.", type=bool, default=False)

arg = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu)

import gym

import trickster as T

env = gym.make(arg.env)

on_policy = {"REINFORCE": T.agent.REINFORCE,
             "A2C": T.agent.A2C,
             "PPO": T.agent.PPO}

off_policy_discreete = {"DQN": T.agent.DQN,
                        "DoubleDQN": T.agent.DoubleDQN}
off_policy_continuous = {"DDPG": T.agent.DDPG,
                         "TD3": T.agent.TD3,
                         "SAC": T.agent.SAC}

available_algos = {}
available_algos.update(on_policy)
available_algos.update(off_policy_discreete)
available_algos.update(off_policy_continuous)
available_algos.update({k.lower(): v for k, v in available_algos.items()})

if arg.algo not in available_algos:
    raise NotImplementedError(f"This algorithm is not implemented: {arg.algo}")

if isinstance(env.action_space, gym.spaces.Box):
    off_policy_discreete_algo_names = set(off_policy_discreete)
    off_policy_discreete_algo_names.update({k.lower() for k in off_policy_discreete})
    if arg.algo in off_policy_discreete_algo_names:
        raise RuntimeError(f"{arg.algo} cannot be used with continuous action spaces.")

if isinstance(env.action_space, gym.spaces.Discrete):
    off_policy_continuous_algo_names = set(off_policy_continuous)
    off_policy_continuous_algo_names.update({k.lower() for k in off_policy_continuous})
    if arg.algo in off_policy_continuous_algo_names:
        raise RuntimeError(f"{arg.algo} cannot be used with discreete action spaces.")

algo = available_algos[arg.algo].from_environment(env)

if arg.algo in on_policy:
    rollout = T.rollout.Trajectory(algo, env, arg.max_steps)
    batch_size = arg.batch_size if arg.algo.lower() == "ppo" else -1

    callbacks = [T.callbacks.ProgressPrinter(rollout.history_keys)]
    if arg.render_frequency > 0:
        callbacks.append(T.callbacks.TrajectoryRenderer(rollout, frequency=arg.render_frequency))

    rollout.fit(arg.train_epochs,
                rollouts_per_epoch=arg.parallel_envs,
                update_batch_size=batch_size,
                callbacks=callbacks)
    if arg.render_final:
        rollout.render(repeats=100)
else:
    test_env = gym.make(arg.env)
    test_rollout = T.rollout.Trajectory(algo, test_env, arg.max_steps)

    if arg.parallel_envs == 1:
        rollout = T.rollout.Rolling(algo, env, arg.max_steps)
    else:
        rollout = T.rollout.MultiRolling(algo, [gym.make(arg.env) for _ in range(arg.parallel_envs)], arg.max_steps)

    callbacks = [T.callbacks.TrajectoryEvaluator(test_rollout, repeats=3),
                 T.callbacks.ProgressPrinter(rollout.history_keys)]
    if arg.render_frequency > 0:
        callbacks.append(T.callbacks.TrajectoryRenderer(test_rollout, frequency=arg.render_frequency))

    rollout.fit(arg.train_epochs,
                updates_per_epoch=32,
                steps_per_update=1,
                update_batch_size=arg.batch_size,
                testing_rollout=test_rollout,
                buffer_warmup=True)

    if arg.render_final:
        test_rollout.render(repeats=100)
