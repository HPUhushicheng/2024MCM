import json
import numpy as np
''''模拟生成10000个3维坐标，以json格式输出，可用于贝叶斯训练  '''
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

# 将结果保存为JSON格式
output_json = json.dumps(submarine_coordinates)

# 输出JSON格式数据
print(output_json)
