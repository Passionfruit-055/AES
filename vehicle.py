import os
import numpy as np
import random

from Crypto.Random import get_random_bytes

from AES import *
from node import Node, random_msg


class Vehicle(Node):
    def __init__(self, index, pos, vel, tx_power, rx_power, road):
        super().__init__(pos, tx_power, rx_power, True)
        self.id = index
        self.name = f"Vehicle{self.id}"
        self.pos = pos
        self.vel = vel
        self.road = road

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


if __name__ == '__main__':
    v = Vehicle(1, [1, 2], 5, 0, 0)
    msg = None
    secrets = v.encrypt_msg(msg if msg is not None else random_msg())
    v.decrypt_msg(secrets)
