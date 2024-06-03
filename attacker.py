from collections import deque
import random

from AES import SUPPORTED_WORK_MODES, SUPPORTED_KEY_LENGTHS


class Attacker(object):
    def __init__(self, args):
        self.args = args

        self.key_length = [16, 24, 32]
        self.work_mode = ['ECB', 'CBC', 'CTR', 'OFB', 'CFB']

        self.max_malicious_msgs = 1000
        self.malicious_msg_num = 1000
        self.length_based_freqs = [0.3, 0.45, 0.6]  # attack freq
        self.mode_based_freq = [0.05 * (i + 1) for i in range(len(self.work_mode))]

        self.msgs = deque(maxlen=args.n_steps)

        self.atk_succ_times = 0
        self.timestep = 0
        self.atk_succ_prob = self.atk_succ_times / (self.timestep + 1)
        self.recent_atk = False
        self.reset()

        self.decrypt_probs = {'mode': [0.95, 0.9, 0.85, 0.65, 0.6], 'length': [0.9, 0.8, 0.7]}
        self.decrypt_prob = random.choice(self.decrypt_probs['mode']) * random.choice(self.decrypt_probs['length'])

    def reset(self):
        self.timestep = 50
        self.atk_succ_times = 25
        self.recent_atk = False
        self.atk_succ_prob = self.atk_succ_times / (self.timestep + 1)
        self.msgs.clear()

    def attack(self, key_length, work_mode):
        self.timestep += 1
        key = self.key_length.index(key_length)
        mode = self.work_mode.index(work_mode)

        self.atk_succ_prob = 1 - (self.length_based_freqs[key] + self.mode_based_freq[mode])
        self.malicious_msg_num = self.max_malicious_msgs * self.atk_succ_prob

        # self.atk_succ_prob = self.decrypt_probs['mode'][mode] * self.decrypt_probs['length'][key]
        #
        # self.recent_atk = False
        # if random.random() < self.atk_succ_prob:
        #     self.atk_succ_times += 1
        #     self.recent_atk = True

        return self.atk_succ_prob, self.malicious_msg_num

    @property
    def atk_succ_rate(self):
        return self.atk_succ_times / (self.timestep + 1)
