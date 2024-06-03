import math
import os
import random
import base64


def random_msg():
    # msgs = {'control': 1000, 'traffic': 1000, 'video': 2000, 'road': 2000}
    msgs = {'control': 1000, 'traffic': 1100, 'video': 1200, 'road': 1300}
    kind = random.choice(list(msgs.keys()))
    random_bytes = os.urandom(msgs[kind])
    base64_encoded = base64.b64encode(random_bytes)
    base64_message = base64_encoded.decode('utf-8')
    # print(f"Msg (Byte): {random_bytes}")
    # print(f"Msg (Base64): {base64_encoded}")
    # print(type(base64_encoded))
    # print(f"Msg (UTF-8): {base64_message}")
    # print(type(base64_message))
    return base64_message


def shannon_capacity(bandwidth, signal_power_dbm, noise_power):
    signal_power_mw = 10 ** (signal_power_dbm / 10)  # dBm转换为毫瓦（mW）
    noise_power_mw = 10 ** (noise_power / 10)  # dBm转换为毫瓦（mW）
    snr = signal_power_mw / noise_power_mw
    capacity = bandwidth * math.log2(1 + snr)
    return capacity

