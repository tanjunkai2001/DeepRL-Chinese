"""3.6节实验环境（LQR控制）。
"""

import gym
import numpy as np
from scipy import linalg

# 线性化CartPole系统参数
g = 9.8
m = 0.1  # 杆子质量
M = 1.0  # 小车质量
l = 0.5  # 杆子长度（实际为一半）
dt = 0.02

# 连续时间线性化系统
A = np.array([
    [0, 1, 0, 0],
    [0, 0, -(m * g) / M, 0],
    [0, 0, 0, 1],
    [0, 0, (M + m) * g / (M * l), 0]
])
B = np.array([
    [0],
    [1 / M],
    [0],
    [-1 / (M * l)]
])

# LQR权重
Q = np.diag([10, 1, 10, 1])
R = np.array([[0.001]])

# 求解连续时间Algebraic Riccati方程
P = linalg.solve_continuous_are(A, B, Q, R)
K = np.dot(np.linalg.inv(R), np.dot(B.T, P))  # 状态反馈增益

env = gym.make("CartPole-v0", render_mode="human")
state, _ = env.reset()

for t in range(1000):
    env.render()
    print(state)

    # LQR控制器
    x = np.array(state)
    u = -np.dot(K, x)
    # CartPole环境动作是离散的，u>0向右，u<0向左
    action = int(u > 0)

    state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Finished")
        state, _ = env.reset()

env.close()
