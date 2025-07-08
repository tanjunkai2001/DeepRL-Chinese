"""3.6节实验环境。
"""


import gym


env = gym.make("CartPole-v0", render_mode="human") # 创建CartPole环境， render_mode="human"表示渲染到屏幕上
state = env.reset() # 重置环境，返回初始状态


for t in range(1000):
    env.render() # 渲染环境
    print(state) # 打印当前状态

    action = env.action_space.sample() # 随机选择一个动作

    state, reward, terminated, truncated, info = env.step(action) # 执行动作，返回新的状态、奖励、是否终止、是否截断和额外信息

    if terminated or truncated:
        print("Finished")
        state = env.reset()

env.close()
