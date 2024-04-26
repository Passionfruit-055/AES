import os
import numpy as np
import random

from node import *


class BaseStation(Node):
    def __init__(self, index, pos, tx_power, rx_power):
        super().__init__(pos, tx_power, rx_power, False)
        self.id = index
        self.name = f"BaseStation{self.id}"
        self.pos = pos
        self.road = [[0, 20], [0, 10]]

