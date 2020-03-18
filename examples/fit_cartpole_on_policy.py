import gym

import trickster as tt

ENV_NAME = "CartPole-v1"
ALGO = "A2C"
TRAJECTORY_MAX_STEPS = 200
EPOCHS = 1000
UPDATES_PER_EPOCH = 64
ROLLOUTS_PER_UPDATE = 4

env = gym.make(ENV_NAME)

algo = {"REINFORCE": tt.agent.REINFORCE,
        "A2C": tt.agent.A2C,
        "PPO": tt.agent.PPO}[ALGO]

agent = algo.from_environment(env)
trajectory = tt.rollout.Trajectory(agent, env, TRAJECTORY_MAX_STEPS)

callbacks = [
    tt.callbacks.ProgressPrinter(trajectory.progress_keys),
    tt.callbacks.TrajectoryRenderer(trajectory, output_to_screen=True, output_files_directory="default"),
    tt.callbacks.TensorBoard(experiment_name=trajectory.experiment_name)
]

trajectory.fit(epochs=EPOCHS,
               updates_per_epoch=UPDATES_PER_EPOCH,
               rollouts_per_update=ROLLOUTS_PER_UPDATE,
               callbacks=callbacks)

trajectory.render(repeats=10)
