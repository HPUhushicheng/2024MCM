import numpy as np
import matplotlib.pyplot as plt

# 目标函数
# def objective_function(x):
#     return x[0]*14903.4 + x[1]*23355.1 + x[2]*12627.6 

#目标函数2
def objective_function(x):
    return x[0]*1031 + x[1]*1755.9 +x[2]*770.3

# 约束条件
def constraint1(x):
    return x[0] + x[1] + x[2] - 5


def constraint2(x):
    return x[0] 

def constraint3(x):
    return  x[1] 

def constraint4(x):
    return x[2] 

# 定义问题的上下界
bounds = [(0, 5)] * 5

# 定义蒙特卡洛模拟的次数
num_samples = 1000

# 初始化目标函数值列表
values = []

# 执行蒙特卡洛模拟
for _ in range(num_samples):
    # 生成随机解
    x = np.random.uniform(0, 5, size=5)
    
    # 检查是否满足约束条件
    if (constraint1(x) <= 0 and constraint2(x) > 0 and constraint3(x) > 0 and constraint4(x) > 0):
        # 计算目标函数值
        value = objective_function(x)
        
        # 添加到目标函数值列表中
        values.append(value)

# 绘制散点图
# 这段代码将每次随机得到的目标函数值绘制成散点图，横坐标为迭代次数，纵坐标为目标函数值。通过观察散点图，可以了解到目标函数值的分布情况。
plt.scatter(range(len(values)), values, s=5, color='#845EC2')
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Monte Carlo Simulation Results')
plt.grid(True)
plt.show()