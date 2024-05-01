import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from IoV import *
from common.arguments import *
from agents.agents import Agents
from agents.DQN_agent import DQN


def set_seed(seed=0, rand=False):
    seed = seed if not rand else random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # # 以下语句会因确定性而损失运行速度
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main():
    args = get_common_args()
    set_seed(args.seed)

    env = IoV(args.n_vehicles, args.n_base_stations, args.n_attackers, args)

    agent = DQN(gamma=args.gamma, lr=args.lr, action_num=len(SUPPORTED_KEY_LENGTHS) * len(SUPPORTED_WORK_MODES),
                state_num=args.n_vehicles * args.state_dim, buffer_size=args.buffer_size, batch_size=args.batch_size,
                INITIAL_EPSILON=0.2, FINAL_EPSILON=0.01, max_episode=args.n_episodes, replace=args.replace)

    rewards_s = []
    atk_succ_times = []
    latencies = []  # 认为每通信 500 次，为一个 episode

    for episode in range(args.n_episodes):
        rewards = []
        latencies = []
        atk_succ_time = 0
        for t in range(args.n_steps):

            state = env.state

            actions, _ = agent.choose_action(state)
            for vehicle in env.vehicles:
                vehicle.set_crypt(actions)

            env.step()

            reward, latency, safe_level = env.compute_reward()
            rewards.append(reward)
            latencies.append(latency)

            next_state = env.state

            agent.store_transition(state, actions, reward, next_state)
            agent.learn()

        rewards_s.append(sum(rewards))
        atk_succ_times.append(atk_succ_time)
        latencies.append(sum(latencies))

    for data, ylabel in zip([rewards_s, atk_succ_times, latencies], ['Reward', 'Attack Success Times', 'Latency']):
        plt.figure()
        plt.plot(data)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.show()


if __name__ == '__main__':
    main()
