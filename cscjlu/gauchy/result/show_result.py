import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
将实验结果从控制台日志手动输入到此，作图展示
"""
resnet34 = [97.44, 97.65, 99.07, 98.11, 99.25,
            99.19, 99.21, 99.24, 99.38, 99.25,
            99.25, 99.38, 99.34, 99.32, 99.28,
            99.32, 99.38, 99.39, 99.39, 99.37]

resnet18 = [89.58, 96.21, 96.90, 97.67, 97.61,
            96.76, 98.47, 98.41, 98.59, 98.91,
            98.21, 98.88, 98.82, 98.87, 98.65,
            98.76, 99.16, 99.17, 99.22, 99.06]

resnet_fix_16 = [93.06, 97.4, 97.33, 97.53, 98.64,
                 97.24, 98.82, 98.96, 99.0, 98.84,
                 99.1, 99.24, 99.05, 99.21, 99.18,
                 99.27, 99.2, 99.14, 99.25, 99.27]

# 训练精度
acc_data = np.array([resnet_fix_16, resnet18, resnet34]).T
df_acc = pd.DataFrame(
    data=acc_data,
    columns=['ResNetFix-16', 'ResNet-18', 'ResNet-34'])
# df_acc.to_excel("acc_data.xlsx")

xticks = [x for x in range(20)]
xticklabels = [x for x in range(1, 21)]
# 创建 Figure 和 Axes 对象
fig, ax = plt.subplots()
# 绘制每个数据系列的折线图
for column in df_acc.columns:
    ax.plot(df_acc.index, df_acc[column], label=column)
# 添加 x 轴和 y 轴标签，以及图表标题
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy %')
ax.set_title('Accuracy vs. Epoch')

# 添加图例
ax.legend()
plt.legend(loc="lower right", fontsize=12)

# 设置坐标轴标签字体大小和颜色
plt.xticks(xticks, xticklabels,
           rotation=-25, size=10, color='grey')
plt.ylim(85, 100)
plt.yticks(size=14, color='grey')
plt.show()

##
rn34 = [datetime.timedelta(seconds=414, microseconds=182753),
        datetime.timedelta(seconds=416, microseconds=644896),
        datetime.timedelta(seconds=426, microseconds=909939),
        datetime.timedelta(seconds=443, microseconds=919553),
        datetime.timedelta(seconds=445, microseconds=219582),
        datetime.timedelta(seconds=446, microseconds=150045),
        datetime.timedelta(seconds=450, microseconds=112622),
        datetime.timedelta(seconds=441, microseconds=338888),
        datetime.timedelta(seconds=442, microseconds=297106),
        datetime.timedelta(seconds=440, microseconds=802610),
        datetime.timedelta(seconds=440, microseconds=309091),
        datetime.timedelta(seconds=439, microseconds=588401),
        datetime.timedelta(seconds=437, microseconds=872245),
        datetime.timedelta(seconds=436, microseconds=455169),
        datetime.timedelta(seconds=438, microseconds=344932),
        datetime.timedelta(seconds=441, microseconds=77437),
        datetime.timedelta(seconds=440, microseconds=345795),
        datetime.timedelta(seconds=441, microseconds=552564),
        datetime.timedelta(seconds=437, microseconds=891190),
        datetime.timedelta(seconds=439, microseconds=349393)]
rn18 = [datetime.timedelta(seconds=412, microseconds=369248),
        datetime.timedelta(seconds=411, microseconds=106749),
        datetime.timedelta(seconds=411, microseconds=469141),
        datetime.timedelta(seconds=411, microseconds=219478),
        datetime.timedelta(seconds=411, microseconds=407194),
        datetime.timedelta(seconds=410, microseconds=999488),
        datetime.timedelta(seconds=410, microseconds=320509),
        datetime.timedelta(seconds=409, microseconds=279872),
        datetime.timedelta(seconds=409, microseconds=774912),
        datetime.timedelta(seconds=410, microseconds=488870),
        datetime.timedelta(seconds=410, microseconds=74539),
        datetime.timedelta(seconds=409, microseconds=549470),
        datetime.timedelta(seconds=409, microseconds=396064),
        datetime.timedelta(seconds=409, microseconds=802804),
        datetime.timedelta(seconds=409, microseconds=390180),
        datetime.timedelta(seconds=409, microseconds=457001),
        datetime.timedelta(seconds=410, microseconds=279069),
        datetime.timedelta(seconds=409, microseconds=707980),
        datetime.timedelta(seconds=409, microseconds=822927),
        datetime.timedelta(seconds=409, microseconds=458393)]
rnf16 = [datetime.timedelta(seconds=396, microseconds=222335),
         datetime.timedelta(seconds=395, microseconds=128508),
         datetime.timedelta(seconds=397, microseconds=370855),
         datetime.timedelta(seconds=393, microseconds=572558),
         datetime.timedelta(seconds=377, microseconds=175626),
         datetime.timedelta(seconds=395, microseconds=294271),
         datetime.timedelta(seconds=394, microseconds=33340),
         datetime.timedelta(seconds=397, microseconds=922922),
         datetime.timedelta(seconds=393, microseconds=624634),
         datetime.timedelta(seconds=395, microseconds=701462),
         datetime.timedelta(seconds=399, microseconds=732879),
         datetime.timedelta(seconds=394, microseconds=279427),
         datetime.timedelta(seconds=393, microseconds=276947),
         datetime.timedelta(seconds=392, microseconds=152582),
         datetime.timedelta(seconds=394, microseconds=451411),
         datetime.timedelta(seconds=389, microseconds=155185),
         datetime.timedelta(seconds=392, microseconds=648483),
         datetime.timedelta(seconds=393, microseconds=561591),
         datetime.timedelta(seconds=393, microseconds=409318),
         datetime.timedelta(seconds=399, microseconds=219479)]

# 训练时间
train_time = np.array([
                     [x.seconds for x in rnf16],
                     [x.seconds for x in rn18],
                     [x.seconds for x in rn34]
]).T

df_train_time = pd.DataFrame(
    data=train_time,
    columns=['ResNetFix-16', 'ResNet-18', 'ResNet-34'])
# df_train_time.to_excel("train_time.xlsx")
xticks = [x for x in range(20)]
xticklabels = [x for x in range(1, 21)]
# 创建 Figure 和 Axes 对象
fig, ax = plt.subplots()
# 绘制每个数据系列的折线图
for column in df_train_time.columns:
    ax.plot(df_train_time.index, df_train_time[column], label=column)
# 添加 x 轴和 y 轴标签，以及图表标题
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Time (seconds)')
ax.set_title('Train Time vs. Epoch')

# 添加图例
ax.legend()
plt.legend(loc="lower right", fontsize=12)

# 设置坐标轴标签字体大小和颜色
plt.xticks(xticks, xticklabels,
           rotation=-25, size=10, color='grey')
plt.ylim(350, 470)
plt.yticks(size=14, color='grey')
plt.show()

train_time_mean = train_time.mean(axis=0)
