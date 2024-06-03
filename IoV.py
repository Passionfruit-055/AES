import numpy as np

from vehicle import Vehicle
from base_station import BaseStation
from node import random_msg
from AES import *
from attacker import Attacker

# TRANSMIT_POWER = [3, 2, 1]  # w
CHANNEL_GAIN = [3e-5, 1e-5, 1e-6, 2e-7, 1e-7]


# NOISE_POWER = [1e-14, 1e-13]  # pw, 这里是平方后的值了


class IoV(object):
    def __init__(self, vehicle_num, bs_num, attacker_num, args):
        self.args = args

        self.vehicle_num = vehicle_num
        self.base_station_num = bs_num
        self.attacker_num = attacker_num

        self.bandwidth = 20 * int(1e6)  # Hz, 20MHz
        # max for vehicle = 26dBm, max for base station = 29dBm
        self.tx_power = {'vehicle': 26, 'base_station': 29}  # dBm
        self.rx_power = self.tx_power
        self.noise_power = -114  # dBm

        self.road = [[0, 750], [0, 30]]  # RSU density city = 500m highway = 1000m
        self.max_vehicle_velocity = 60  # km/h

        self.vehicles = [
            Vehicle(index=i + 1, pos=[random.uniform(self.road[0][0], self.road[0][1]),
                                      random.uniform(self.road[1][0], self.road[1][1])],
                    vel=[random.uniform(0, self.max_vehicle_velocity), random.uniform(0, self.max_vehicle_velocity)],
                    tx_power=random.randint(self.tx_power['vehicle'] - 5, self.tx_power['vehicle']),
                    gain=random.choice(CHANNEL_GAIN), road=self.road)
            for i in range(vehicle_num)]
        self.base_stations = [
            BaseStation(index=i + 1,
                        pos=[random.uniform(self.road[0][0], self.road[0][1]), random.choice(self.road[1])],
                        tx_power=self.tx_power['base_station'],
                        rx_power=self.rx_power['base_station']) for i in range(bs_num)]
        self.attackers = [Attacker(args) for _ in range(attacker_num)]

        self.msgs = {}
        self.keys = {}

    def reset(self):
        self.msgs = {}
        self.keys = {}

        for vehicle in self.vehicles:
            vehicle.reset()

        for attacker in self.attackers:
            attacker.reset()

    def send(self, timestep, ready2send=None):
        sending_seq = ready2send if ready2send is not None else self.vehicles

        trans_latency = 0
        crypt_latency = 0

        msg = None
        key_length = None
        work_mode = None

        for node in sending_seq:  # 这里只用 index 0
            msg = random_msg()
            key_length = node.key_length
            work_mode = node.work_mode
            secrets, iv, nonce = node.encrypt_msg(msg)
            self.keys[node.name] = node.key
            self.msgs[node.name] = secrets
            # 加解密时延
            raw_msg_length = get_string_bit_length(msg)
            encoding_latency_v = node.crypt_latency(raw_msg_length, timestep)
            crypt_latency = max(crypt_latency, encoding_latency_v)
            # 传输时延
            secrets_length = get_string_bit_length(secrets)
            tx_latency_v = node.trans_latency(secrets_length, self.bandwidth, self.noise_power, timestep)
            trans_latency = max(trans_latency, tx_latency_v)

        return crypt_latency, trans_latency, msg, key_length, work_mode

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

    def step(self, timestep):
        for vehicle in self.vehicles:
            vehicle.move()

        encoding_latency, tx_latency, msg, key_length, work_mode = self.send(timestep)
        atk_succ_probs = []
        malicious_msg_nums = []
        for attacker in self.attackers:
            atk_succ_prob, malicious_msg_num = attacker.attack(key_length, work_mode)
            malicious_msg_nums.append(malicious_msg_num)
            atk_succ_probs.append(atk_succ_prob)
        # decoding_latency = self.receive()
        # print(f"Encoding latency: {encoding_latency}")
        # print(f"Transmit latency: {tx_latency}")
        # print(f"Total latency: {(encoding_latency * 2 + tx_latency * 2) * 1e3} ms")
        return np.mean(atk_succ_probs), np.mean(malicious_msg_nums)

    @property
    def state(self):
        s = []
        one_hots = np.eye(self.vehicle_num).tolist()
        for vehicle, one_hot in zip(self.vehicles, one_hots):
            s.append(vehicle.state + one_hot)
        return s

    def compute_reward(self):
        rewards = []
        for vehicle in self.vehicles:
            rewards.append(vehicle.reward)

        defense_levels = [vehicle.defense_level for vehicle in self.vehicles]
        avg_defense_level = np.mean(defense_levels)

        trans_lates = [vehicle.latency['transmit'] for vehicle in self.vehicles]
        trans_latency = np.max(trans_lates)

        encrypt_lates = [vehicle.latency['crypt'] for vehicle in self.vehicles]
        encrypt_latency = np.max(encrypt_lates)

        latency = trans_latency + encrypt_latency

        return np.mean(rewards), rewards, latency, avg_defense_level


if __name__ == '__main__':
    from common.arguments import get_common_args
    args = get_common_args()
    env = IoV(2, 1, 1, args)
    s_tate = env.state
    print(s_tate)
