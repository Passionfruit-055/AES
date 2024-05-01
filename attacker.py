from collections import deque

import numpy
import random


class Attacker(object):
    def __init__(self, args):
        self.args = args
        self.launch_atk_probs = [0.3, 0.5, 0.8, ]  # attack freq
        self.decrypt_probs = [0.8, 0.7, 0.6, 0.5, 0.4]  # safe level, mode > length

        self.key_length = [16, 24, 32]
        self.work_mode = ['ECB', 'CBC', 'CTR', 'OFB', 'CFB']
        self.msgs = deque(maxlen=args.n_steps)
        self.atk_succ_times = 0
        self.timestep = 0

        self.launch_atk_prob = random.choice(self.launch_atk_probs)
        self.decrypt_prob = random.choice(self.decrypt_probs)
        self.atk_succ_prob = 1

    def attack(self, msg, key_length, work_mode):
        key = self.key_length.index(key_length)
        mode = self.work_mode.index(work_mode)

        atk_prob = self.launch_atk_prob[key]
        decrypt_prob = self.decrypt_prob[mode]

        self.atk_succ_prob = atk_prob * decrypt_prob

        if random.random() < self.atk_succ_prob:
            self.atk_succ_times += 1
            return True
        else:
            return False

    @property
    def atk_succ_rate(self):
        return self.atk_succ_times / self.timestep
