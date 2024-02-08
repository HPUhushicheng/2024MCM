import numpy as np
import matplotlib.pyplot as plt

def simulate_submarine_position(A, B, u, initial_state, dt, num_steps):
    """模拟潜水器的定位"""
    # 初始化位置和速度数组
    positions = np.zeros((num_steps,))
    velocities = np.zeros((num_steps,))
    
    # 设置初始状态
    x = initial_state
    
    # 迭代模拟
    for i in range(num_steps):
        # 计算下一个时间步长的状态
        x = np.dot(A, x) + np.dot(B, u)
        
        # 存储位置和速度
        positions[i] = x[0]
        velocities[i] = x[1]
    
    return positions, velocities

# 示例参数
dt = 1  # 时间步长
num_steps = 1000  # 模拟步数
m = 1000 # 潜水器质量
Fb = 2000 # 浮力
Fg = 9800 # 重力
Fc = 500 # 水流作用力
Fd = 500
# 初始状态
initial_state = np.array([0, 0])  # 初始位置和速度都为零

# 获取动力学方程的矩阵形式
A = np.array([[1, dt],
              [0, 1]])
B = np.array([[0, 0, 0, 0],
              [dt / m, 0, 0, 0]])
u = np.array([Fb, Fg, Fc, Fd])

# 模拟潜水器的定位
positions, velocities = simulate_submarine_position(A, B, u, initial_state, dt, num_steps)

# 绘制潜水器位置随时间的变化图
time = np.arange(0, num_steps * dt, dt)
plt.plot(time, positions)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Submarine Position Over Time')
plt.grid(True)
plt.show()

time = np.arange(0, num_steps * dt, dt)
plt.plot(time, velocities)
plt.xlabel('Time')
plt.ylabel('velocities')
plt.title('Submarine Speed Over Time')
plt.grid(True)
plt.show()
