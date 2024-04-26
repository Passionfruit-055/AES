import os
import numpy as np
import random
from vehicle import Vehicle
from base_station import BaseStation
from node import random_msg
from AES import *

import math


class IoV(object):
    def __init__(self, vehicle_num, bs_num, args):
        self.args = args
        self.vehicle_num = vehicle_num
        self.base_station_num = bs_num

        self.bandwidth = 20 * int(1e6)  # Hz, 20MHz
        # max for vehicle = 26dBm, max for base station = 29dBm
        self.tx_power = {'vehicle': 23, 'base_station': 25}  # dBm
        self.rx_power = self.tx_power
        self.noise_power = -114  # dBm

        self.road = [[0, 750], [0, 30]]  # RSU density city = 500m highway = 1000m
        self.max_vehicle_velocity = 60  # km/h

        self.vehicles = [
            Vehicle(index=i, pos=[random.uniform(self.road[0][0], self.road[0][1]),
                                  random.uniform(self.road[1][0], self.road[1][1])],
                    vel=random.uniform(0, self.max_vehicle_velocity),
                    tx_power=self.tx_power['vehicle'], rx_power=self.rx_power['vehicle'], road=self.road)
            for i in range(vehicle_num)]
        self.base_stations = [
            BaseStation(index=i, pos=[random.uniform(self.road[0][0], self.road[0][1]), random.choice(self.road[1])],
                        tx_power=self.tx_power['base_station'],
                        rx_power=self.rx_power['base_station']) for i in range(bs_num)]
        self.msgs = {}
        self.keys = {}

    def reset(self):
        self.msgs = {}
        self.keys = {}

        for vehicle in self.vehicles:
            vehicle.move()

    def send(self, ready2send=None):
        sending_seq = ready2send if ready2send is not None else self.vehicles
        tx_latency = 0
        encoding_latency = 0
        for node in sending_seq:
            msg = random_msg()
            secrets = node.encrypt_msg(msg)
            self.keys[node.name] = node.key
            self.msgs[node.name] = secrets
            # 加密时延
            raw_msg_length = get_string_bit_length(msg)
            encoding_latency_v = node.crypt_latency(raw_msg_length, 'encrypt')
            encoding_latency = max(encoding_latency, encoding_latency_v)
            # 传输时延
            secrets_length = get_string_bit_length(secrets)
            tx_latency_v = node.trans_latency(secrets_length, self.bandwidth, self.noise_power)
            tx_latency = max(tx_latency, tx_latency_v)
        return encoding_latency, tx_latency

    def receive(self, ready2receive=None):
        receiving_seq = ready2receive if ready2receive is not None else self.base_stations
        decoding_latency = 0
        for node in receiving_seq:
            for sender in self.msgs.keys():
                if sender == node.name:
                    continue
                else:
                    secrets = self.msgs[sender]
                    key = self.keys[sender]
                    msg = node.decrypt_msg(secrets, key)
                    # 解密时延
                    secrets_length = get_string_bit_length(secrets)
                    decoding_latency_v = node.crypt_latency(secrets_length, 'decrypt')
                    decoding_latency = max(decoding_latency, decoding_latency_v)
        return decoding_latency

    def set_key_length(self, lengths):
        assert len(lengths) == self.vehicle_num, 'Lengths must be equal to the number of vehicles'
        for vehicle, length in zip(self.vehicles, lengths):
            vehicle.set_key_length(length)

    def step(self):
        encoding_latency, tx_latency = self.send()
        decoding_latency = self.receive()
        print(f"Encoding latency: {encoding_latency}")
        print(f"Transmit latency: {tx_latency}")
        print(f"Decoding latency: {decoding_latency}")
        print(f"Total latency: {encoding_latency + tx_latency + decoding_latency}")


if __name__ == '__main__':
    env = IoV(5, 5, None)
    env.step()
