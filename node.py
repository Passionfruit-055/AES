import os
import numpy as np
import base64
import random

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from AES import *
import math


def random_msg():
    msgs = {'control': 100, 'traffic': 50, 'video': 200, 'road': 300}
    kind = random.choice(list(msgs.keys()))
    random_bytes = os.urandom(msgs[kind])
    base64_encoded = base64.b64encode(random_bytes)
    base64_message = base64_encoded.decode('utf-8')
    # print(f"Msg (Byte): {random_bytes}")
    # print(f"Msg (Base64): {base64_encoded}")
    # print(type(base64_encoded))
    print(f"Msg (UTF-8): {base64_message}")
    # print(type(base64_message))
    return base64_message


def shannon_capacity(bandwidth, signal_power_dbm, noise_power):
    signal_power_mw = 10 ** (signal_power_dbm / 10)  # dBm转换为毫瓦（mW）
    noise_power_mw = 10 ** (noise_power / 10)  # dBm转换为毫瓦（mW）
    snr = signal_power_mw / noise_power_mw
    capacity = bandwidth * math.log2(1 + snr)
    return capacity


class Node(object):
    def __init__(self, pos, tx_power, rx_power, movable, supported_protocol=None):
        self.pos = pos
        self.tx_power = tx_power
        self.rx_power = rx_power
        self.movable = movable
        self.supported_protocol = ['WIFI', 'LTE', 'Bluetooth'] if supported_protocol is None else supported_protocol
        self.possible_key_length = (16, 24, 32)  # bytes
        self.key_length = 32
        self.latency = {'encrypt': 0, 'decrypt': 0, 'transmit': 0}
        self.main_freq = [1.7 * int(1e9), 2.84 * int(1e9)]  # GHz 每秒多少个时钟周期
        self.bits_per_cycle = 64  # bytes 64位机器，每一个周期能处理 64 位宽
        self.compute_capacity = self.main_freq[0] * self.bits_per_cycle  # bps

    def set_key_length(self, length):
        if length in self.possible_key_length:
            self.key_length = length
        else:
            raise ValueError(f"Key length must be one of {self.possible_key_length}")

    @property
    def key(self):
        return get_random_bytes(self.key_length)
        # key = str(base64.encodebytes(key), encoding='utf-8')

    def encrypt_msg(self, msg):
        return AES_encrypt(msg, self.key)

    def decrypt_msg(self, secret_str, sender_key):
        return AES_decrypt(secret_str, sender_key)

    def trans_latency(self, length, bandwidth, noise_power):
        self.latency['transmit'] = length / shannon_capacity(bandwidth, self.tx_power, noise_power)
        return self.latency['transmit']

    def crypt_latency(self, length, mode):
        self.latency[mode] = length / self.compute_capacity
        return self.latency[mode]
