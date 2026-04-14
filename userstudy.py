import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 准备数据 ---
data = {
    "Guidance Following": [4.708, 3.769, 1.827, 1.679, 1.345, 1.554, 1.628],
    "Style Consistency": [4.554, 3.449, 2.542, 3.244, 2.536, 2.631, 2.511],
    "Identity Preservation": [4.596, 3.605, 2.321, 3.161, 2.673, 2.558, 2.242]
}

index = [
    "Ours",
    "Image\nSculpting",
    "Anydoor",
    "Drag\nDiffusion",
    "Drag\nAnything",
    "DiffEditor",
    "InstantDrag"
]
df = pd.DataFrame(data, index=index)

# --- 2. 设置学术论文风格 ---
# 使用中等饱和度的配色(色盲友好,更有活力)
colors = ['#E8B17A', '#7DB3D5', '#6BAA96']  # 中等饱和度的橙、蓝、绿

# 设置高质量字体(LaTeX风格)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,  # 增大基础字体
    'text.usetex': False,
    'axes.labelsize': 16,  # 增大轴标签
    'axes.titlesize': 17,
    'xtick.labelsize': 20,  # 进一步增大X轴刻度标签
    'ytick.labelsize': 15,
    'legend.fontsize': 14,  # 增大图例字体
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
})

# --- 3. 创建图表(学术标准尺寸)---
fig, ax = plt.subplots(figsize=(14, 4.8), facecolor='white', dpi=300)  # 增宽图表
ax.set_facecolor('white')  # 纯白背景

# 计算柱子位置
x = np.arange(len(index))
width = 0.26  # 柱子宽度（新增类目后略微减小）

# 绘制柱状图(添加细黑边框)
bars1 = ax.bar(x - width, df.iloc[:, 0], width, label='Guidance Following', 
               color=colors[0], edgecolor='black', linewidth=0.8, 
               hatch='', alpha=0.90)  # 调整透明度
bars2 = ax.bar(x, df.iloc[:, 1], width, label='Style Consistency', 
               color=colors[1], edgecolor='black', linewidth=0.8,
               hatch='', alpha=0.90)
bars3 = ax.bar(x + width, df.iloc[:, 2], width, label='Identity Preservation', 
               color=colors[2], edgecolor='black', linewidth=0.8,
               hatch='', alpha=0.90)

# --- 4. 设置坐标轴 ---
ax.set_ylabel('Average Score (1-5)', fontsize=16, fontweight='normal')

# X轴设置
ax.set_xticks(x)
ax.set_xticklabels(index, fontsize=18, ha='center')

# Y轴设置(刻度更稀疏)
ax.set_ylim(0, 5.2)
ax.set_yticks(np.arange(0, 6, 1))  # 只显示0, 1, 2, 3, 4, 5

# 添加细腻的网格线(仅y轴)
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.15, color='black')
ax.set_axisbelow(True)

# --- 5. 优化边框和刻度 ---
# 保留所有边框但使用细线
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('black')

# 刻度线设置
ax.tick_params(axis='both', which='major', direction='out', 
               length=4, width=1.2, colors='black')
ax.tick_params(axis='both', which='minor', direction='out', 
               length=2, width=1.0, colors='black')

# --- 6. 图例优化(学术风格)---
legend = ax.legend(loc='upper right', frameon=True, 
                   edgecolor='black', fancybox=False,
                   framealpha=1.0, ncol=1, columnspacing=1.0,
                   handlelength=1.5, handleheight=0.7)
legend.get_frame().set_linewidth(1.0)

# --- 7. 添加数值标签(放在柱子内部)---
def autolabel(bars, rotation=0):
    for bar in bars:
        height = bar.get_height()
        # 将数字放在柱子内部,从底部向上偏移一定距离
        y_pos = height * 0.5  # 放在柱子中间位置
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.2f}',
                ha='center', va='center', fontsize=12,
                rotation=rotation, color='#2C3E50', fontweight='bold')

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# --- 8. 调整布局 ---
plt.tight_layout(pad=0.3)

# --- 9. 保存和显示 ---
plt.savefig('userstudy_results.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none', format='pdf')
plt.savefig('userstudy_results.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()