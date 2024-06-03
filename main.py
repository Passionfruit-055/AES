import random
import numpy as np
import torch
import scipy.io as sio
import pickle

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

plt.style.use('bmh')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['text.color'] = 'black'
plt.set_cmap('jet')
colors = ['blue', 'orange', 'red', 'forestgreen', 'orange', 'darkviolet', ]

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from IoV import *
from common.arguments import *
from agents.agents import Agents
from agents.DQN_agent import DQN
from agents.Fixed_work_mode import FixedAgent
from agents.VDN import VDN


# def end_test():
#     env.close()
#     if args.log:
#         print('saving final data set')
#         pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
#         pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))
#         if base_method == 'sac':
#             torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
#         else:
#             torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
#             pickle.dump(model_optim.log, open(path + 'optim_data'+ '.pkl', 'wb'))
#
#         # save duration
#         end = datetime.now()
#         date_str = end.strftime("%Y-%m-%d_%H-%M-%S/")
#         duration_str = get_duration(start_time)
#
#         # save config
#         with open(path + "/../config.txt","a") as f:
#             f.write('End Time\n')
#             f.write('\t'+ date_str + '\n')
#             f.write('Duration\n')
#             f.write('\t'+ duration_str + '\n')
#             f.close()
#
#         # save final steps
#         if args.pointmass:
#             fig_saved = False
#             try:
#                 if args.render:
#                     if args.singleshot:
#                         viewer.save(path + "/final_fig_viewer.svg")
#                         fig_saved = True
#                     viewer.close()
#             except:
#                 pass
#             if not fig_saved:
#                 try:
#                     traj.save_fig(path + "/final_fig.svg")
#                 except:
#                     traj.save_buff(path + "/final_fig.pkl")
#         else:
#             buff = replay_buffer.get_final_samples(10000)
#             pickle.dump(buff, open(path + 'buffer_data'+ '.pkl', 'wb'))

# def handler(signal_received, frame):
#     # Handle any cleanup here
#     print('SIGINT or CTRL-C detected.')
#     end_test()
#     print('Exiting gracefully')
#     exit(0)


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


def smooth(data, sm=5):  # 平滑曲线
    z = np.ones(len(data))
    y = np.ones(sm) * 1.0
    smooth_data = np.convolve(y, data, "same") / np.convolve(y, z, "same")
    return smooth_data


AGENTS_MODES = {'VDN': 16, 'IQL': 15, 'CFB-32': 14, 'ECB-16': 0}


def load_agent(args, mode, ):
    if mode == 'IQL':
        agent = DQN(gamma=args.gamma, lr=args.lr,
                    action_num=len(SUPPORTED_KEY_LENGTHS) * len(SUPPORTED_WORK_MODES),
                    state_num=args.state_dim, buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    INITIAL_EPSILON=0.2, FINAL_EPSILON=0.01, max_episode=args.n_episodes,
                    replace=args.replace)
    elif mode == 'VDN':
        agent = VDN(gamma=args.gamma, lr=args.lr,
                    action_num=len(SUPPORTED_KEY_LENGTHS) * len(SUPPORTED_WORK_MODES),
                    state_num=args.state_dim, buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    INITIAL_EPSILON=0.2, FINAL_EPSILON=0.01, max_episode=args.n_episodes,
                    replace=args.replace)
    else:
        agent = FixedAgent(args, AGENTS_MODES[mode])

    return agent


def _store_transition(agent, mode, state, action, reward, state_, ):
    if mode == 'IQL':
        for s, a, r, ns in zip(state, action, reward, state_):
            agent.store_transition(s, a, r, ns)
    elif mode == 'VDN':
        agent.store_transition(state, action, reward, state_)
    else:
        pass


def main():
    args = get_common_args()
    args.state_dim = args.state_dim + args.n_vehicles
    env = IoV(args.n_vehicles, args.n_base_stations, args.n_attackers, args)

    mode_results = {'reward': {}, 'latency': {}, 'protection_level': {}, 'atk_succ_rate': {}}
    for mode in AGENTS_MODES.keys():

        print(f'Work in {mode.upper()} mode')

        rewards_s_avg = []
        atk_succ_times_avg = []
        latencies_s_avg = []
        protection_levels_s_avg = []
        malicious_msgs_s_avg = []

        n_avg_times = args.n_avg_times if mode in ['VDN', 'IQL'] else 1

        for avg_time in tqdm(range(n_avg_times)):

            agent = load_agent(args, mode)

            set_seed(args.seed, True)

            rewards_s = []
            atk_succ_times = []
            latencies_s = []
            protection_levels_s = []
            malicious_msgs_s = []

            for episode in tqdm(range(args.n_episodes)):

                rewards = []
                latencies = []
                protection_levels = []
                atk_succ_time = []
                malicious_msgs = []
                next_state = None
                env.reset()

                # one transmit = one episode
                for t in range(args.n_steps):

                    state = env.state if t == 0 else next_state

                    actions = []
                    for vehicle, s in zip(env.vehicles, state):
                        a, _ = agent.choose_action(s)
                        actions.append(a)
                        vehicle.set_crypt(a)

                    atk_succ_prob, malicious_msg_time = env.step(t)

                    avg_malicious_msg = int(np.mean(malicious_msg_time))
                    for v in env.vehicles:
                        v.under_attack(avg_malicious_msg)

                    reward_avg, reward_agents, latency, protection_level = env.compute_reward()
                    rewards.append(reward_avg)
                    latencies.append(latency)
                    protection_levels.append(protection_level)
                    atk_succ_time.append(atk_succ_prob)
                    malicious_msgs.append(avg_malicious_msg)

                    next_state = env.state

                    _store_transition(agent, mode, state, actions, reward_agents, next_state)
                    agent.learn()

                rewards_s.append(sum(rewards))
                latencies_s.append(sum(latencies))
                atk_succ_times.append(np.mean(atk_succ_time))
                protection_levels_s.append(sum(protection_levels) / (len(protection_levels) + 1))
                malicious_msgs_s.append(malicious_msgs)

            rewards_s_avg.append(rewards_s)
            atk_succ_times_avg.append(atk_succ_times)
            latencies_s_avg.append(latencies_s)
            protection_levels_s_avg.append(protection_levels_s)
            malicious_msgs_s_avg.append(malicious_msgs_s)

        rewards_s_avg = np.mean(rewards_s_avg, axis=0).tolist()
        atk_succ_times_avg = np.mean(atk_succ_times_avg, axis=0).tolist()
        latencies_s_avg = np.mean(latencies_s_avg, axis=0).tolist()
        protection_levels_s_avg = np.mean(protection_levels_s_avg, axis=0).tolist()
        malicious_msgs_s_avg = np.mean(malicious_msgs_s_avg, axis=0).tolist()

        if mode in ['IQL', 'VDN']:
            for data, ylabel in zip(
                    [rewards_s_avg, latencies_s_avg, protection_levels_s_avg, atk_succ_times_avg],
                    ['Reward', 'Latency (ms)', 'Protection level', 'Attack success rate (%)']):
                plt.figure()
                plt.plot(data)
                plt.xlabel('Episode')
                plt.ylabel(ylabel)
                plt.tight_layout()
                plt.savefig(f'./results/{mode}_{ylabel.lower()}.png')
                plt.title(mode.title())
                plt.show()
                plt.close()
            plt.close()

        for key, value in zip(['reward', 'latency', 'protection_level', 'atk_succ_rate'],
                              [rewards_s_avg, latencies_s_avg, protection_levels_s_avg, atk_succ_times_avg]):
            mode_results[key][mode] = value

    sio.savemat(f'./results/mat/all_results.mat', mode_results)

    for (key, value), ylabel in zip(mode_results.items(),
                                    ['Reward', 'Latency (ms)', 'Protection level', 'Attack success rate (%)']):
        plt.figure()
        for (mode, data), color in zip(value.items(), colors):
            plt.plot(smooth(data) if mode == 'IQL' else data, label=mode, color=color)
        plt.xlabel('Episode')
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./results/all_{key}.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
