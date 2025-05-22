from learningAgent import QL_Agent
import numpy as np



class LearningLoop:
    def __init__(self, env):
        max_epsilon = 1.0
        min_epsilon = 0.1
        decay_rate = 0.0005
        self.agent = QL_Agent(env, discount_factor=0.95, learning_rate=0.1, epsilon=max_epsilon)

        total_episodes = 50000
        rewards = np.zeros((total_episodes))
        for i in range (total_episodes):
            total_rewards = 0
            s, _ = env.reset()
            while True:
                a = self.agent.action(s)
                s_, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                self.agent.updateQTable(s, a, r, s_)
                s = s_
                total_rewards += r

                self.agent.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * i)
                if done:
                    break
            rewards[i] = total_rewards

        print("Final Q-Table")
        print(self.agent.Qtable)