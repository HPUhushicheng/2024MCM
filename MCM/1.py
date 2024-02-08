import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def read_asc_file(file_path):
    """读取ASC文件，并返回地形数据"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # 找到NODATA_value所在的行
        nodata_index = [i for i, line in enumerate(lines) if 'NODATA_value' in line][0]
        
        # 从NODATA_value所在的下一行开始读取数据
        data = [list(map(float, line.split())) for line in lines[nodata_index+1:]]
        data = np.array(data)
        
        # 获取行数和列数
        nrows, ncols = data.shape
        
        # 从ASCII文件中获取X、Y、Z坐标
        x = np.arange(0, ncols, 1)
        y = np.arange(0, nrows, 1)
        x, y = np.meshgrid(x, y)
        z = data[::-1]  # 将数据反转，使得坐标原点位于左下角
        return x, y, z


def plot_3d_terrain(x, y, z):
    """绘制3D地形图"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('')
    ax.set_title('3D Terrain Map')
    plt.show()

# ASC文件路径
asc_file_path = 'gebco_2023.asc'

# 读取ASC文件
x, y, z = read_asc_file(asc_file_path)

# 绘制3D地形图
plot_3d_terrain(x, y, z)
