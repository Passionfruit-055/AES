import random

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from AES import *
from utils.communication import *

SUPPORTED_MSG_TYPE = ['control', 'traffic', 'video', 'gps']


class Node(object):
    def __init__(self, pos, tx_power, gain, movable, index):
        self.comm_data = None
        self.index = index
        self.pos = pos
        self.movable = movable

        self.possible_key_length = [16, 24, 32]  # bytes
        self.key_length = random.choice(self.possible_key_length)

        self.possible_work_mode = SUPPORTED_WORK_MODES
        self.work_mode = random.choice(SUPPORTED_WORK_MODES)

        self.possible_msg_type = SUPPORTED_MSG_TYPE
        self.msg_type = random.choice(SUPPORTED_MSG_TYPE)

        self.estimate_malicious_msg = 1000
        self.malicious_msg_num = 0

        self.defense_level = 1

        self.latency = {'crypt': 0.0, 'transmit': 0.0, 'encrypt': 0.0, 'decrypt': 0.0, 'tx': 0.0, 'rx': 0.0}
        self.all_latency = {'transmit': [], 'encrypt': [], 'decrypt': []}
        # crypt
        self.main_freq = [1.7 * int(1e9), 2.84 * int(1e9)]  # GHz 每秒多少个时钟周期
        self.bits_per_cycle = 64  # bytes 64位机器，每一个周期能处理 64 位宽
        self.compute_capacity = self.main_freq[0] * self.bits_per_cycle  # bps
        # transmit
        self.bandwidth = 0
        self.tx_power = tx_power
        self.gain = gain
        self.noise_power = -114

        self._load_latency()

    def reset(self):
        self.latency = {'crypt': 0.0, 'transmit': 0.0, 'encrypt': 0.0, 'decrypt': 0.0, 'tx': 0.0, 'rx': 0.0}
        self.malicious_msg_num = 0

    def update_channel(self, gain, tx_power, bandwidth=None, noise_power=None, ):
        if bandwidth is not None:
            self.bandwidth = bandwidth
        if noise_power is not None:
            self.noise_power = noise_power
        self.gain = gain
        self.tx_power = tx_power

    def _set_key_length(self, length):
        if length in self.possible_key_length:
            self.key_length = length
        else:
            raise ValueError(f"Key length must be one of {self.possible_key_length}")

    def _set_work_mode(self, mode):
        if mode in self.possible_work_mode:
            self.work_mode = mode
        else:
            raise ValueError(f"Work mode must be one of {self.possible_work_mode}")

    @property
    def key(self):
        return get_random_bytes(self.key_length)
        # key = str(base64.encodebytes(key), encoding='utf-8')

    # @property
    # def risk_level(self):
    #     return (self.key_length / 8) + SUPPORTED_WORK_MODES.index(self.work_mode) + 1  # 密钥长度比工作模式影响的权重更大

    def encrypt_msg(self, msg):
        return AES_encrypt(msg, self.key, self.work_mode)

    def decrypt_msg(self, secret_str, sender_key):
        return AES_decrypt(secret_str, sender_key)

    def trans_latency(self, length, bandwidth, noise_power, timestep):
        # # simulate
        # self.latency['transmit'] = (length / 10) / shannon_capacity(bandwidth, self.tx_power * self.gain, noise_power)
        # real
        self.latency['transmit'] = self.get_trans_latency(timestep, length)
        return self.latency['transmit']

    def crypt_latency(self, msg_length, timestep):
        # # simulate
        # mode_weight = 0.8 + (SUPPORTED_WORK_MODES.index(self.work_mode) + 1) / 5
        # self.latency['crypt'] = mode_weight * self.key_length * msg_length * 10 / self.compute_capacity
        # real
        self.latency['crypt'] = self.get_crypt_latency(self.work_mode, self.key_length, timestep, msg_length / 1600)
        return self.latency['crypt']

    def set_crypt(self, action):
        length = action // len(self.possible_work_mode)
        length = self.possible_key_length[int(length)]
        self._set_key_length(length)

        mode = action % len(self.possible_work_mode)
        mode = self.possible_work_mode[int(mode)]
        self._set_work_mode(mode)

        # print(f"Set key length to {length}, work mode to {mode}")

    def _load_latency(self):
        from raw_data.extract import _crypt, _iw_xlsx
        self.all_latency['crypt'] = _crypt(self.index)
        self.comm_data = _iw_xlsx(self.index)

    def get_crypt_latency(self, mode, key_len, timestep, prop=1):
        # print(f"{mode}-{key_len}")
        size = 100
        try:
            self.latency['encrypt'] = self.all_latency['crypt'][f'EN_{mode}_{key_len}_{size}'][timestep]
            self.latency['decrypt'] = self.all_latency['crypt'][f'DE_{mode}_{key_len}_{size}'][timestep]
            # print(f"encrypt:{round(self.latency['encrypt'], 2)} ms")
            # print(f"decrypt:{round(self.latency['decrypt'], 2)} ms")
            return (self.latency['encrypt'] + self.latency['decrypt']) * prop
        except KeyError:
            raise ValueError(f"No {self.index} latency data for {mode}_{key_len}_{size} ")

    def get_trans_latency(self, timestep, size=1000):
        try:
            self.latency['tx'] = size / (self.comm_data['tx bitrate (MBit)'][timestep] * int(1e6))
            self.latency['rx'] = size / (self.comm_data['rx bitrate (MBit)'][timestep] * int(1e6))
            # print(f"tx latency:{round(self.latency['tx'], 6)} ms")
            # print(f"rx latency:{round(self.latency['rx'], 6)} ms")
            return self.latency['tx'] + self.latency['rx']
        except KeyError:
            raise ValueError(f"No latency data for {size} at {timestep}")


if __name__ == '__main__':
    pass
    # n_node = 5
    # nodes = [Node(pos=[0, 0], tx_power=26, gain=0, movable=True, index=i) for i in range(1, 1 + n_node)]
    # for node in nodes:
    #     node.load_latency()
    # print('Loading done')
    # for node in nodes:
    #     for t in range(1000):
    #         node.get_crypt_latency(random.choice(SUPPORTED_WORK_MODES), random.choice(SUPPORTED_KEY_LENGTHS), t)
    #         node.get_trans_latency(t, random.randint(1000, 2000))
