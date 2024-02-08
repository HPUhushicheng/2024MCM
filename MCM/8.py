import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''随机模拟100只潜水器的坐标,并3d展示'''

def simulate_submarine_coordinates(num_submarines, duration, step_size):
    coordinates = []
    for i in range(num_submarines):
        x = np.random.uniform(0, 100)  # 模拟x坐标
        y = np.random.uniform(0, 100)  # 模拟y坐标
        z = np.random.uniform(0, 100)  # 模拟z坐标
        submarine_path = []
        for t in range(duration):
            x += np.random.normal(0, 0.1)  # 添加随机扰动模拟航行
            y += np.random.normal(0, 0.1)
            z += np.random.normal(0, 0.1)
            submarine_path.append([x, y, z])
        coordinates.append(submarine_path)
    return coordinates

# 模拟参数
num_submarines = 100  # 潜水器数量
duration = 100  # 模拟持续时间
step_size = 1  # 时间步长

# 模拟潜水器坐标
submarine_coordinates = simulate_submarine_coordinates(num_submarines, duration, step_size)

# 3D可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for submarine_path in submarine_coordinates:
    submarine_path = np.array(submarine_path)
    ax.plot(submarine_path[:, 0], submarine_path[:, 1], submarine_path[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
