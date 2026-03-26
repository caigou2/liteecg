import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据 (请确保文件名和路径正确)
df = pd.read_csv('each_F1.xlsx - Sheet1.csv')

# 提取模型名称和分类
models = df.iloc[1:, 0].values.tolist()
categories = ['N', 'SVEB', 'VEB', 'Other']

# 提取 Stra 和 Rand 的 F1 数据 (转换为浮点数)
stra_data = np.array([
    df.iloc[1:, 1].astype(float).values,
    df.iloc[1:, 2].astype(float).values,
    df.iloc[1:, 3].astype(float).values,
    df.iloc[1:, 4].astype(float).values
])

rand_data = np.array([
    df.iloc[1:, 5].astype(float).values,
    df.iloc[1:, 6].astype(float).values,
    df.iloc[1:, 7].astype(float).values,
    df.iloc[1:, 8].astype(float).values
])

# 2. 设置您指定的 RGB 配色，并转换为 Hex 格式
colors_rgb = [
    (151, 153, 152), (198, 146, 135), (231, 154, 144),
    (239, 188, 145), (228, 205, 135), (250, 229, 184),
    (221, 221, 223)
]
# 将RGB转换成十六进制颜色码供matplotlib使用
colors = ['#%02x%02x%02x' % rgb for rgb in colors_rgb]

# 3. 初始化图表 (1行2列的并排子图)
fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=300, sharey=True)
# 缩小两个子图之间的间距
fig.subplots_adjust(wspace=0.05) 

x = np.arange(len(categories))
n_models = len(models)
total_width = 0.85
width = total_width / n_models

# 为模仿参考图，我们在第2个(SVEB)和第4个(Other)类别加上斜线阴影纹理
hatches = ['', '\\\\', '', '\\\\']

# 定义在柱子上方添加数值标签的函数
def add_labels(rects, ax):
    for rect in rects:
        height = rect.get_height()
        if height > 0:  # 只有当高度大于0时才添加标签
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移3个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90) # 旋转90度防止7个模型数字重叠

# 4. 绘制柱状图
for i in range(n_models):
    # 计算每个模型对应柱子的X轴位置
    pos = x - total_width/2 + i*width + width/2
    
    # 按照类别(N, SVEB, VEB, Other)逐个绘制，以便分别应用不同阴影(hatch)
    for c in range(len(categories)):
        # 左侧图：Stra
        rect_stra = axes[0].bar(pos[c], stra_data[c, i], width, 
                                color=colors[i], edgecolor='black', zorder=3, hatch=hatches[c])
        add_labels(rect_stra, axes[0])
        
        # 右侧图：Rand (只在第一个分类时添加label参数，防止图例重复)
        label = models[i] if c == 0 else ""
        rect_rand = axes[1].bar(pos[c], rand_data[c, i], width, 
                                color=colors[i], edgecolor='black', zorder=3, hatch=hatches[c], label=label)
        add_labels(rect_rand, axes[1])

# 5. 图表美化与参数调整
axes[0].set_title('MIT-BIH-Stra', fontsize=16, fontweight='bold', pad=15)
axes[1].set_title('MIT-BIH-Rand', fontsize=16, fontweight='bold', pad=15)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    # 添加水平虚线网格线，放在底层(zorder=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    # 隐藏上方和右方的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 因为存在0.0的数据，我们不截断Y轴，统一展示0-1.1范围
    ax.set_ylim(0, 1.1)

# 在右侧图表外侧添加图例
axes[1].legend(loc='upper right', frameon=True, fontsize=11, bbox_to_anchor=(1.15, 1))

# 6. 保存或展示
plt.savefig('result_bar_chart.pdf', bbox_inches='tight') # 推荐保存为PDF矢量图以插入论文
plt.show()