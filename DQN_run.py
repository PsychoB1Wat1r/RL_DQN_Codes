import random
import gym
from DQN_brain import DQN, ReplayBuffer
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from Model.DRL_Learning.Advanced import rl_utils
import time  # 在每个迭代末尾添加一个小延迟，便于观察结果
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 加载实验环境
env_name = 'CartPole-v1'
env = gym.make("CartPole-v1", render_mode="human")

lr = 2e-3  # 学习率
num_episodes = 500  # 回合数
hidden_dim = 128  # 隐含层神经元个数
gamma = 0.9  # 贪心系数
epsilon = 0.1  # 贪心策略系数
target_update = 200  # 目标网络更新频率
buffer_size = 10000  # 经验池容量
minimal_size = 500  # 经验池超过minimal_size后再训练
batch_size = 64  # 一次所取得的经验池数据个数
return_list = []  # 记录每个回合（episodes）的回报
# 设置随机种子 通过设置种子，程序每次运行时生成的随机数序列都是相同的，这对于调试和实验复现非常重要。
random.seed(0)  # 设置Python标准库的随机种子
np.random.seed(0)  # 设置NumPy的随机种子
torch.manual_seed(0)  # 设置PyTorch的随机种子
state = env.reset(seed=0)  # 对于Gym >= 0.19创建Gym环境并设置随机种子
env.seed(0)  # 如果使用较旧版本的Gym，可以使用env.seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
replay_buffer = ReplayBuffer(buffer_size)  # 实例化经验池
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)  # 实例化DQN

# 训练过程
for i in range(10):  # 迭代十轮共打印10条pbar 在每一轮的迭代中进行num_episodes/10个回合
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0  # 记录每个回合的回报
            state, _ = env.reset()  # 每个回合开始前重置环境 产生一个随机状态
            done = False
            while not done:
                action = agent.take_action(state)  # 获取当前状态下需要采取的动作
                next_state, reward, done, _, _ = env.step(action)  # 更新环境 奖励 终止符
                replay_buffer.add(state, action, reward, next_state, done)  # 收集数据
                state = next_state  # 更新当前状态
                episode_return += reward  # 计算当前回合的累计回报
                if replay_buffer.size() > minimal_size:  # 判断缓存池数据量是否达到训练要求
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d
                    }
                    agent.update(transition_dict)  # 训练模型
            return_list.append(episode_return)  # 记录每个回合的回报 用于绘图
            if (i_episode + 1) % 10 == 0:  # 在控制台打印回合数和累计回报信息
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)  # 单次回合结束后更新进度条1/50-->2/50...45/50-->50/50 用于动态显示进度
            time.sleep(0.01)  # 可选：打印输出延迟，便于观察

# 绘图部分保持不变
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()