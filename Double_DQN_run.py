import random
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from Double_DQN_brain import DQN
from Model.DRL_Learning.Advanced import rl_utils
import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 加载实验环境
env_name = 'CartPole-v1'
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
# 实验参数
learning_rate = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.9
epsilon = 0.1
target_update = 100
buffer_size = 1000
minimal_size = 100
batch_size = 64
# 设置随机种子 通过设置种子，程序每次运行时生成的随机数序列都是相同的，这对于调试和实验复现非常重要。
random.seed(0)  # 设置 Python 标准库的随机种子
np.random.seed(0)  # 设置 NumPy 的随机种子
torch.manual_seed(0)  # 设置 PyTorch 的随机种子
state = env.reset(seed=0)  # 对于 Gym >= 0.19 创建 Gym 环境并设置随机种子
env.seed(0)  # 如果使用较旧版本的 Gym，可以使用 env.seed(0)

# # 加载环境
# env_name = 'Pendulum-v1'
# env = gym.make("Pendulum-v1", render_mode="human")
# state_dim = env.observation_space.shape[0]
# action_dim = 11  # 将连续动作分成11个离散动作

# def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
#     action_lowbound = env.action_space.low[0]  # 连续动作的最小值
#     action_upbound = env.action_space.high[0]  # 连续动作的最大值
#     return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)
def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):  # 定义训练过程的函数
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):  # 迭代十轮共打印10条pbar 在每一轮的迭代中进行num_episodes/10个回合
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 记录每个回合episode的回报
                state, _ = env.reset()  # 每个回合开始前重置环境 产生一个随机状态 对于 gym >= 0.26
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    # action_continuous = dis_to_con(action, env, agent.action_dim)
                    # next_state, reward, done, _, _ = env.step([action_continuous])
                    next_state, reward, done, _, _ = env.step(action)  # 更新环境 奖励 终止符
                    replay_buffer.add(state, action, reward, next_state, done)  # 收集数据
                    state = next_state  # 更新当前状态
                    episode_return += reward  # 计算当前回合的累计回报
                    if replay_buffer.size() > minimal_size:  # 判断缓存池数据量是否达到训练要求
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
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
    return return_list, max_q_value_list


# DQN训练
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device)
return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

# Double-DQN训练
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, 'DoubleDQN')
return_list, max_q_value_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()

frames_list = list(range(len(max_q_value_list)))
plt.plot(frames_list, max_q_value_list)
plt.axhline(0, c='orange', ls='--')
plt.axhline(10, c='red', ls='--')
plt.xlabel('Frames')
plt.ylabel('Q value')
plt.title('Double DQN on {}'.format(env_name))
plt.minorticks_on()
plt.grid(which='minor', linestyle='--', linewidth=0.75)
plt.show()
