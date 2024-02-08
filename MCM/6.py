

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 模拟潜水器在三维空间中的运动轨迹
t = np.linspace(0, 20, 1000)
x = t * np.sin(t)
y = t * np.cos(t)
z = np.log(t + 1)

# 模拟另一潜水器为对比
x2 = t * np.sin(t + 1)
y2 = t * np.cos(t + 1)
z2 = np.log(t + 0.5)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制潜水器的运动轨迹
ax.plot(x, y, z, label='Submersible 1 Trajectory', color='blue', linewidth=2)
ax.plot(x2, y2, z2, label='Submersible 2 Trajectory', color='red', linestyle='--', linewidth=2)

# 标记起点和终点
ax.scatter(x[0], y[0], z[0], color='green', s=100, label='Start Point', edgecolor='black')
ax.scatter(x[-1], y[-1], z[-1], color='orange', s=100, label='End Point', edgecolor='black')

# 设置图形属性
ax.set_title('3D Trajectories of Submersibles in the Sea', fontsize=20)
ax.set_xlabel('X Axis', fontsize=14)
ax.set_ylabel('Y Axis', fontsize=14)
ax.set_zlabel('Z Axis (Depth)', fontsize=14)
ax.legend(fontsize=12)

plt.show()