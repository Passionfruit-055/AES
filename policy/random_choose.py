import numpy as np
import random
import torch


class RandomChoose(object):
    def __init__(self, n_actions, args):
        self.n_actions = n_actions
        self.last_action = 0
        self.args = args

    def choose_action(self, q_value=None, avail_actions=None, epsilon=None):
        self.last_action = np.random.choice(range(self.n_actions))
        return self.last_action

    def get_q_value(self, q_value, avail_actions):
        pass

    def learn(self, batch):
        pass

    def learn_step(self, batch):
        pass

    def get_params(self):
        return None

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass


if __name__ == '__main__':
    # 定义可选集
    options = [1, 2, 3, 4, 5]
