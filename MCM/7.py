import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# 生成模拟数据
# 生成主船路径
ship_path = np.random.randint(0, 10, size=(100, 2))
# 生成潜水器路径
submarine_path = np.random.randint(0, 10, size=(100, 2))

# 构建特征和标签
X = np.concatenate([ship_path, submarine_path])
y = np.array([0] * len(ship_path) + [1] * len(submarine_path))  # 0表示主船，1表示潜水器

# 训练朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(X, y)

# 可视化展示
plt.figure(figsize=(10, 6))

# 绘制主船路径
plt.plot(ship_path[:, 0], ship_path[:, 1], 'ro-', label='Ship Path')
# 绘制潜水器路径
plt.plot(submarine_path[:, 0], submarine_path[:, 1], 'bo-', label='Submarine Path')

# 构建网格来绘制决策边界
xx, yy = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.title('Path Relationship between Ship and Submarine')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()
