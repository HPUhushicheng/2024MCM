import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(1, 100, 100)  # 在1~100s内采样100次
a = 0.6  
v0 = 0  
s0 = 0  
imu_var = 120 ** 2  # IMU测量误差方差，Q~N(0, imu_var)
gps_var = 50 ** 2  # GPS测量误差方差，R~N(0, gps_var)
num_samples = t.shape[0]

# 根据理想模型推导出来的真实位置值
real_positions = [0] * num_samples
real_positions[0] = s0
# 模拟观测值，通过理论值加上观测噪声模拟获得
measure_positions = [0] * num_samples
measure_positions[0] = real_positions[0] + np.random.normal(0, gps_var ** 0.5)
# 不使用卡尔曼滤波，也不使用实际观测值修正，单纯依靠运动模型来预估的预测值
predict_positions = [0] * num_samples
predict_positions[0] = measure_positions[0]
# 最优估计值，也就是卡尔曼滤波输出的真实值的近似逼近
optim_positions = [0] * num_samples
optim_positions[0] = measure_positions[0]
# 卡尔曼滤波算法的中间变量
pos_k_1 = optim_positions[0]

predict_var = 0
for i in range(1, t.shape[0]):
    # 根据理想模型获得当前的速度、位置真实值
    real_v = v0 + a * i
    real_pos = s0 + (v0 + real_v) * i / 2
    real_positions[i] = real_pos
    # 模拟输入数据，实际应用中从传感器测量获得
    v = real_v + np.random.normal(0, imu_var ** 0.5)
    measure_positions[i] = real_pos + np.random.normal(0, gps_var ** 0.5)
    # 如果仅使用运动模型来预测整个轨迹，而不使用观测值，则得到的位置如下
    predict_positions[i] = predict_positions[i - 1] + (v + v + a) * (i - (i - 1)) / 2
    # 以下是卡尔曼滤波的整个过程
    # 根据实际模型预测，利用上个时刻的位置（上一时刻的最优估计值）和速度预测当前位置
    pos_k_pred = pos_k_1 + v + a / 2
    # 更新预测数据的方差
    predict_var += gps_var
    # 求得最优估计值
    pos_k = pos_k_pred * imu_var / (predict_var + imu_var) + measure_positions[i] * predict_var / (
                predict_var + imu_var)
    # 更新
    predict_var = (predict_var * imu_var) / (predict_var + imu_var)
    pos_k_1 = pos_k
    optim_positions[i] = pos_k

plt.plot(t, real_positions, label='hypothetical model positions')
#plt.plot(t, measure_positions, label='measured positions')
plt.plot(t, optim_positions, label='kalman filtered positions')
plt.plot(t, predict_positions, label='predicted positions')
plt.legend()
plt.show()
