import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from main import smooth, colors

all_results = sio.loadmat('./results/mat/all_results.mat')
algos = ['DQN', 'ECB-16', 'CFB-32']
for performance, y_label in zip(['reward', 'latency', 'risk_level', 'atk_succ_times'], ['Reward', 'Latency (ms)', 'Safe level', 'Attack success probability (%)',]):
    plt.figure()
    data = all_results[performance]['DQN'].tolist()[0][0].tolist()[0]
    data = smooth(data)
    plt.plot(data)
    plt.title('DQN')
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'./results/mat/DQN_{performance}.png')
    # plt.show()


with open('./results/mat/all_results.txt', 'w') as f:
    for performance, y_label in zip(['reward', 'latency', 'risk_level', 'atk_succ_times'], ['Reward', 'Latency (ms)', 'Safe level', 'Attack success probability (%)',]):
        plt.figure()
        print(y_label)
        f.write(f'\n{y_label}\n')
        for algo, color in zip(algos, colors):
            data = all_results[performance][algo].tolist()[0][0].tolist()[0]
            data = smooth(data)
            print(f'{algo}:{np.mean(data[-50:])}')
            f.write(f'{algo}:{np.mean(data[-50:])}\n')
            plt.plot(data, label=algo, color=color)
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(f'./results/mat/{performance}.png')
        plt.show()
        plt.close()