import random
import numpy as np
import collections
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 创建一个先进先出的队列，最大长度为capacity

    def add(self, state, action, reward, next_state, done):  # 将数据以元组形式添加进经验池buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,每次取出batch_size个数据，前提是经验回放池数据数量不低于batch_size个
        transitions = random.sample(self.buffer, batch_size)  # 随机抽样数据 防止序贯决策数据间存在相关性（独立性）
        state, action, reward, next_state, done = zip(*transitions)  # 将元组数据“解压” 赋值给五个列表
        #  将状态和下一个状态转换为NumPy数组。这是因为在训练神经网络时，通常需要将数据转换为数组格式，以便批量处理和高效计算。
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):  # 全连接神经网络
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        x = self.fc2(x)
        return x


class DQN:
    """ DQN算法 """
    # .to(device)是PyTorch中一个非常重要的方法，用于将张量（tensors）或模型（models）移动到指定的设备（device）上进行计算。设备通常是CPU或GPU。
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim  # 动作的维度
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器 记录更新次数 用于目标网络的延迟更新
        self.device = device  # 在GPU计算

        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 实例化Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 实例化目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 使用Adam优化器，更新训练网络的参数

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)  # 先将状态转换为batch_size x action_dim维的张量
            action = self.q_net(state).argmax().item()  # 将状态输入至q_net 并找出Q值最大的哪个动作
        return action

    # 训练网络
    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        # print(states.shape), print(states)
        # view()函数是PyTorch中用于重新调整张量形状的方法。参数-1表示自动推断这一维的大小，1表示调整为每个动作占据一列。
        # 这里的作用是将一维的动作张量（形状为（batch_size, ））转换为二维张量（形状为(batch_size, 1)）。这样做通常是为了
        # 匹配后续操作（如gather）所需的维度。
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        # print(actions.shape), print(actions)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # 提取执行指定动作actions后对应的Q值
        # print(q_values.shape), print(q_values)
        # 下个状态的最大Q值：将next_states输入进target_q_net，生成batch_size x action_dim维的数据，调用max(1)[0]函数，
        # 参数1表示沿着列方向（动作方向）进行最大值搜索，[0]是因为max方法返回两个值：最大值和最大值的索引。[0]表示提取最大值，[1]表示提取最大值的索引
        # 这样会得到一个64x1的行向量，即：每个状态执行每个动作后的最大Q值，然后用view(-1, 1)转换为64x1维的张量，用于后续做TD
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 目标网络 防止自举造成偏差传播
        # print(max_next_q_values)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积，这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()  # 对训练网络更新

        if self.count % self.target_update == 0:  # 在一段时间后更新目标网络参数 目标网络的更新频率要低于DQN 防止训练DQN过程中的不稳定
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 将目标网络的参数替换成训练网络的参数
        self.count += 1
