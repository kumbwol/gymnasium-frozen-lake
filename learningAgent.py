import numpy as np

class QL_Agent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.epsilon = epsilon
        self.terminals = [5,7,11,12,15]
        self.env=env

        action_size = env.action_space.n
        state_size = env.observation_space.n
        self.Qtable = np.zeros((state_size, action_size))
        print("Q-table")
        print(self.Qtable)

    def action(self, s):
        exp_exp_tradeoff = np.random.uniform()
        if exp_exp_tradeoff > self.epsilon:
            action = np.argmax(self.Qtable[s,:])
        else:
            action = self.env.action_space.sample()
        return action

    def updateQTable(self, s, a, r, s_):
        self.Qtable[s, a] = self.Qtable[s, a] + self.lr * (r + self.g * np.max(self.Qtable[s_, :]) - self.Qtable[s, a])