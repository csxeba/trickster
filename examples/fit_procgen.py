import procgen
import gym

import trickster as tr

env = gym.make("procgen:procgen-coinrun-v0")
envs = [gym.make("procgen:procgen-coinrun-v0") for _ in range(4)]

agent = tr.agent.PPO.from_environment(env, actor_updates=20, critic_updates=20, entropy_beta=1e-3)
trajectory = tr.rollout.Trajectory(agent, env, max_steps=None)
rolling = tr.rollout.MultiRolling(agent, envs, max_steps=None)

artifactory = tr.utility.artifactory.Artifactory.make_default(experiment_name=rolling.experiment_name)

callbacks = [
    tr.callbacks.LearningRateScheduler.make_exponential(optimizer=agent.actor.optimizer,
                                                        num_epochs=10000,
                                                        start_value=1e-3,
                                                        decay_rate=0.9999,
                                                        min_value=1e-6),
    tr.callbacks.LearningRateScheduler.make_exponential(optimizer=agent.critic.optimizer,
                                                        num_epochs=10000,
                                                        start_value=1e-3,
                                                        decay_rate=0.9999,
                                                        min_value=1e-6),
    tr.callbacks.ProgressPrinter(trajectory.progress_keys),
    tr.callbacks.TensorBoard(artifactory)
]

rolling.fit(epochs=5, updates_per_epoch=32, steps_per_update=32, update_batch_size=-1, warmup_buffer=True,
            callbacks=callbacks)
