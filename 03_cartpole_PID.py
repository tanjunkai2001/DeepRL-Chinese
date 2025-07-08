"""3.6节实验环境。
"""


import gym


env = gym.make("CartPole-v0", render_mode="human") # 创建CartPole环境， render_mode="human"表示渲染到屏幕上
state, _ = env.reset() # 重置环境，返回初始状态


for t in range(1000):
    env.render() # 渲染环境
    print(state) # 打印当前状态

    # PID控制器参数
    if t == 0:
        # 初始化PID控制器变量
        kp = 1.0  # 比例系数
        ki = 0.01  # 积分系数
        kd = 0.1   # 微分系数
        
        prev_error = 0
        integral = 0
    
    # 计算误差（目标是保持杆子垂直，角度为0）
    cart_position = state[0]
    cart_velocity = state[1]
    pole_angle = state[2]
    pole_velocity = state[3]
    
    # 以杆子角度为主要控制目标
    error = pole_angle
    
    # PID计算
    integral += error
    derivative = error - prev_error
    
    # PID输出
    pid_output = kp * error + ki * integral + kd * derivative
    
    # 根据PID输出选择动作（0：向左，1：向右）
    if pid_output > 0:
        action = 1  # 向右
    else:
        action = 0  # 向左
    
    prev_error = error

    state, reward, terminated, truncated, info = env.step(action) # 执行动作，返回新的状态、奖励、是否终止、是否截断和额外信息

    if terminated or truncated:
        print("Finished")
        state, _ = env.reset()

env.close()
