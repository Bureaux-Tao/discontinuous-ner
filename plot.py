##
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm

##
path = "./log/cadec.csv"
df = pd.read_csv(path)
data = [df.F1.values, df.Precision.values, df.Recall.values]

##
plt.rcParams['font.family'] = ['Times']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率

font_zh = fm.FontProperties(fname='/Users/Bureaux/Library/Fonts/TimesSong.ttf')

fig, ax = plt.subplots()  # 创建图实例

dot = ["o", "d", "s"]
label = ["F1", "Precision", "Recall"]

for i in range(len(data)):
    x = np.linspace(0, data[i].shape[0], data[i].shape[0])  # 创建x的取值范围
    # print(x)
    ax.plot(x, data[i], 'k', marker=dot[i], markersize=5, markerfacecolor='k',
            markeredgecolor='k', markevery=1, label=label[i])

ax.set_xlabel('Epochs')  # 设置x轴名称 x label
ax.set_ylabel('Metric')  # 设置y轴名称 y label
ax.set_title('CADEC', fontproperties=font_zh)  # 设置图名为Simple Plot
# plt.figure(figsize = (20, 10))
ax.legend()  # 自动检测要在图例中显示的元素，并且显示
plt.savefig("./assets/cadec.png")
plt.show()
