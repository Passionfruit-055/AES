import math
import os
import numpy as np
import random

from Crypto.Random import get_random_bytes

from AES import *
from node import Node, random_msg, SUPPORTED_MSG_TYPE


class Vehicle(Node):
    def __init__(self, index, pos, vel, tx_power, gain, road):
        super().__init__(pos, tx_power, gain, True, index)
        self.id = index
        self.name = f"Vehicle{self.id}"
        self.pos = pos
        self.vel = vel
        self.road = road

        self.pos_init = pos
        self.vel_init = vel

    def reset(self):
        self.pos = self.pos_init
        self.vel = self.vel_init
        super().reset()

    def move(self):
        direction = random.choice(['keepGoing', 'turnAround'])
        if direction == 'keepGoing':
            self.pos[0] += self.vel[0]
        else:
            self.pos[0] -= self.vel[0]

        dist = random.choice(['near', 'away'])
        if dist == 'near':
            self.pos[1] += self.vel[1]
        else:
            self.pos[1] -= self.vel[1]

        if self.pos[0] < self.road[0][0]:
            self.pos[0] = self.road[0][0]
        elif self.pos[0] > self.road[0][1]:
            self.pos[0] = self.road[0][1]

        if self.pos[1] < self.road[1][0]:
            self.pos[1] = self.road[1][0]
        elif self.pos[1] > self.road[1][1]:
            self.pos[1] = self.road[1][1]

    def under_attack(self, malicious_msg_num, estimate_msg_num=None):
        if estimate_msg_num is not None:
            self.estimate_malicious_msg = estimate_msg_num
        self.malicious_msg_num = malicious_msg_num
        if self.estimate_malicious_msg < self.malicious_msg_num:
            print(f'malicious_msg_num is too large for Vehicle {self.index}')

    @property
    def risk_level(self):
        self.defense_level = math.log2(1 + (self.key_length // 8 - 1) * (self.possible_work_mode.index(self.work_mode) + 1))
        return self.estimate_malicious_msg / self.defense_level

    @property
    def reward(self):
        type_weight = (self.possible_msg_type.index(self.msg_type) + 1) / len(self.possible_msg_type)
        w_T = 1
        w_L = 1e-2
        w_D = 1

        reward = type_weight * self.key_length / 8 - w_T * (self.latency['transmit'] + self.latency[
            'crypt']) - w_L * self.risk_level - w_D * self.malicious_msg_num / self.estimate_malicious_msg

        return reward

    @property
    def state(self):
        return [self.latency['crypt'], self.latency['transmit'], self.gain, self.risk_level, self.pos[0],
                self.pos[1]]


if __name__ == '__main__':
    pass
