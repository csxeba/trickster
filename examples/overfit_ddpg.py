import gym

from trickster.agent import DDPG
from trickster.rollout import Rolling

env = gym.make("Pendulum-v0")

agent = DDPG.from_environment(env,
                              discount_gamma=1.,
                              action_noise_sigma=0.1,
                              action_noise_sigma_decay=1.,
                              action_noise_sigma_min=0.1,
                              polyak_tau=0.01)

rollout = Rolling(agent, env)
rollout.roll(32, verbose=0, learning=True)

agent.actor.optimizer.learning_rate = 0.
agent.critic.optimizer.learning_rate = 1e-4

print(f"Got a batch of {agent.transition_memory.N} transitions")

for epoch in range(1, 11):
    for update in range(1, 10001):
        history = rollout.agent.fit(batch_size=-1)
        print(f"\rEpoch {epoch} progress {update / 100:.0f}% "
              f"- Loss: {history['critic_loss']:.4f} "
              f"- Q: {history['Q']:.4f}", end="")
    print()
