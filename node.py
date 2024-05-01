from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from AES import *
from utils.communication import *


class Node(object):
    def __init__(self, pos, tx_power, gain, movable):
        self.pos = pos
        self.movable = movable

        self.possible_key_length = [16, 24, 32]  # bytes
        self.key_length = random.choice(self.possible_key_length)

        self.possible_work_mode = SUPPORTED_WORK_MODES
        self.work_mode = random.choice(SUPPORTED_WORK_MODES)

        self.latency = {'crypt': 0, 'transmit': 0}

        # crypt
        self.main_freq = [1.7 * int(1e9), 2.84 * int(1e9)]  # GHz 每秒多少个时钟周期
        self.bits_per_cycle = 64  # bytes 64位机器，每一个周期能处理 64 位宽
        self.compute_capacity = self.main_freq[0] * self.bits_per_cycle  # bps

        # transmit
        self.bandwidth = 0
        self.tx_power = tx_power
        self.gain = gain
        self.noise_power = -114

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

    @property
    def safe_level(self):
        return (self.key_length / 8) + SUPPORTED_WORK_MODES.index(self.work_mode) + 1  # 密钥长度比工作模式影响的权重更大

    def encrypt_msg(self, msg):
        return AES_encrypt(msg, self.key, self.work_mode)

    def decrypt_msg(self, secret_str, sender_key):
        return AES_decrypt(secret_str, sender_key)

    def trans_latency(self, length, bandwidth, noise_power):
        self.latency['transmit'] = length / shannon_capacity(bandwidth, self.tx_power * self.gain, noise_power)
        return self.latency['transmit']

    def crypt_latency(self, msg_length):
        mode_weight = 1 + SUPPORTED_WORK_MODES.index(self.work_mode) / 5
        self.latency['crypt'] = mode_weight * self.key_length * msg_length / self.compute_capacity
        return self.latency['crypt']

    def set_crypt(self, action):
        length = action / len(self.possible_key_length)
        length = self.possible_key_length[int(length)]
        self._set_key_length(length)
        mode = action % len(self.possible_key_length)
        mode = self.possible_work_mode[int(mode)]
        self._set_work_mode(mode)
