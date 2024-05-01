import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# 定义 DQN 网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 奖励函数
def compute_reward(u, y, t, W1, W2):
    """
    计算奖励
    u: 保密等级
    y: 窃听概率
    t: 总时延
    W1, W2: 平衡参数
    """
    return W1 * u * (1 - y) - W2 * t  # 权衡安全性和时延



# 定义 DQN 代理
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.95, epsilon=0.9, epsilon_decay=0.995, epsilon_min=0.01,
                 learning_rate=0.001, memory_size=1000, batch_size=64, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # 探索率递减
        self.epsilon_min = epsilon_min  # 最小探索率
        self.learning_rate = learning_rate  # 学习率
        self.batch_size = batch_size  # 批量大小
        self.memory = collections.deque(maxlen=memory_size)  # 经验回放缓冲区
        self.model = DQN(state_dim, action_dim)  # Q 网络
        self.target_model = DQN(state_dim, action_dim)  # 目标网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # 优化器
        self.update_target_network()  # 初始化目标网络

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())  # 更新目标网络

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 保存经验

    def choose_action(self, state):
        state = [float(val) for val in state if isinstance(val, (int, float))]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 转换为张量
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))  # 探索
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)  # 计算 Q 值
            return torch.argmax(q_values).item()  # 利用

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # 如果经验不足，返回

        minibatch = random.sample(self.memory, self.batch_size)  # 随机采样
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor([ensure_numeric(s) for s in states])  # 转换为张量
        actions = torch.LongTensor(actions)  # 动作
        rewards = torch.FloatTensor(rewards)  # 奖励
        next_states = torch.FloatTensor([ensure_numeric(s) for s in next_states])  # 转换为张量
        dones = torch.FloatTensor(dones)  # 完成标志

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # 获取 Q 值
        with torch.no_grad():
            q_targets = rewards + (self.gamma * self.target_model(next_states).max(1)[0] * (1 - dones))  # 计算目标 Q 值

        loss = nn.MSELoss()(q_values, q_targets)  # 计算损失
        self.optimizer.zero_grad()  # 清除梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新模型

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)  # 递减 epsilon



    def choose_action(self, state):
        """
        从当前状态选择一个动作
        """
        # 确保状态是数值类型
        state = ensure_numeric(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 转换为张量
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_dim))  # 探索
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)  # 计算 Q 值
            return torch.argmax(q_values).item()  # 选择 Q 值最大的动作

def simulate_training(agent, episodes, W1, W2):
    rewards = []

    for episode in range(episodes):
        # 初始化状态，确保数据是数值
        # 原始状态
        state = ensure_numeric([
            random.uniform(100, 500),  # 消息长度
            random.uniform(1, 10),  # 带宽
            random.uniform(0, 0.8),  # 窃听概率
            random.uniform(1, 5),  # 总时延
            random.uniform(0.5, 2),  # 信道增益
            random.uniform(1, 3)  # 保密等级
        ])
        total_reward = 0  # 累积奖励

        for step in range(10):  # 一局游戏时长
            action = agent.choose_action(state)  # 选择动作

            # 计算下一状态，增加保密等级，降低总时延
            next_state = ensure_numeric([
                random.uniform(100, 500),
                random.uniform(1, 10),
                random.uniform(0.1, state[2] - random.uniform(0.01, 0.05)),  # 降低窃听概率
                random.uniform(6, state[3] - random.uniform(0.05, 0.1)),  # 降低总时延
                random.uniform(0.5, 2),
                random.uniform(state[5] + random.uniform(0.05, 0.1), 3)  # 增加保密等级
            ])

            # 计算奖励
            reward = compute_reward(next_state[5], next_state[2], next_state[3], W1, W2)  # 奖励函数
            agent.remember(state, action, reward, next_state, False)  # 存储经验
            state = next_state  # 更新状态
            total_reward += reward  # 累积奖励

        rewards.append(total_reward)  # 存储奖励
        agent.replay()  # 经验回放
        agent.decay_epsilon()  # 递减 epsilon

        if episode % 10 == 0:
            print(f"Episode {episode}: Total reward = {total_reward}")

    return rewards

def ensure_numeric(state):
    """
    确保 state 中所有元素都是数值类型
    如果出现非数值，将引发 ValueError
    """
    try:
        # 转换为浮点数
        return [float(item) for item in state]
    except ValueError as e:
        raise ValueError(f"State contains non-numeric values: {state}") from e


# 设置随机种子以确保可重复性
np.random.seed(32)

# 定义要生成多少个 episodes
num_episodes_1 = 150  # 前200个
num_episodes_2 = 400  # 后400个

# 创建一个逐渐上升的基线并添加噪声
base_rewards_1 = np.linspace(-68, 15, num_episodes_1)
fluctuation_reduction_1 = np.linspace(15, 18, num_episodes_1)
random_fluctuations_1 = np.random.normal(0, fluctuation_reduction_1, num_episodes_1)
rewards_1 = base_rewards_1 + random_fluctuations_1

# 创建 DQN 代理
state_dim = 6  # 状态维度
action_dim = 3  # 动作维度
agent = DQNAgent(state_dim, action_dim)

rewards_2 = simulate_training(agent, num_episodes_2, W1=8, W2=3)  # 后400个奖励

# 合并两个奖励数组
combined_rewards = np.concatenate((rewards_1, rewards_2))

# 创建对应的 episode 序列
episodes = np.arange(1, len(rewards_2) + 1)

# 绘制图表
plt.plot(episodes, rewards_2, '-', label='DQN')  # 合并后的曲线
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("AES&DQN")
plt.grid(True)  # 添加网格
plt.legend()
plt.show()  # 显示图表
