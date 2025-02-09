import numpy as np
import torch
import torch.nn.functional as F


class Qnet(torch.nn.Module):  # 继承父类class Module
    """ 只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()  # 父类又称之为超类superclass super()函数让子类可以调用父类的方法
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):  # 前向传播
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """ DQN算法,包括Double DQN """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 原始网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def take_action(self, state):  # epsilon-贪婪策略
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # 提取执行指定动作actions后对应的Q值
        # DQN与Double DQN的区别
        if self.dqn_type == 'DoubleDQN':  # 如果使用的是DDQN 则需要找到q_net最大时的动作max_action  并使用target_q_net来计算在状态next_states时，选取动作max_action时的值
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)  # 防止最大化造成TD目标的高估
        else:  # 如果使用的是DQN 则直接使用target_q_net来计算在状态next_states时最大值 不再要求动作为max_action
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 目标网络 防止自举造成偏差传播
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.count % self.target_update == 0:  # 在一段时间后更新目标网络参数 目标网络的更新频率要低于DQN 防止训练DQN过程中的不稳定
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 将目标网络的参数替换成训练网络的参数
        self.count += 1
