from torch.distributed.elastic import agent
import numpy as np


class PlayTest:
    def __init__(self, env, agent):
        print("kak")
        winAmount = 0
        total_episodes = 50
        rewards = np.zeros((total_episodes))

        for i in range(total_episodes):
            total_rewards = 0
            s, _ = env.reset()

            while True:
                a = agent.action(s)
                s_, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                s = s_
                total_rewards += r
                if done:
                    if s == 15:
                        winAmount += 1
                    break
            rewards[i] = total_rewards

        print("kak", winAmount)
