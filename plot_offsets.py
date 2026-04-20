import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ================= 用户配置区 =================
# 请将这里替换为您刚才生成的测试结果文件夹名称
# 例如: 'CAST_PEMS04_96_96_CAST_PEMS_Climate_ftM_sl96_ll48_pl96_dm64_el2_df256_eb307_abloriginal_0'
SETTING_NAME = 'PEMS04_2018_WithWeather_CAST_PEMS_Climate_ftM_sl96_ll48_pl12_dm64_el2_df2048_eb307_0' 
DATA_PATH = f'test_results/{SETTING_NAME}/offsets_analysis.npy'
# ============================================

def generate_offset_boxplot():
    if not os.path.exists(DATA_PATH):
        print(f"Error: 找不到文件 {DATA_PATH}。请先运行测试脚本生成该文件。")
        return

    # 1. 加载数据
    data = np.load(DATA_PATH, allow_pickle=True).item()
    offsets = data['offsets']
    weathers = data['weathers'] # 形状: [Samples, 5] (One-hot)

    # 2. 定义天气等级名称 (必须与您 fuse_data_level.py 里的顺序严格一致)
    # 假设顺序是: No Rain, Drizzle, Light Rain, Moderate Rain, Heavy Rain
    level_names = ['No Rain', 'Drizzle', 'Light Rain', 'Moderate Rain', 'Heavy Rain']

    # 3. 将 One-Hot 标签还原为文本标签
    weather_labels = []
    for w in weathers:
        # 找到 one-hot 中值为 1 的索引
        idx = np.argmax(w)
        weather_labels.append(level_names[idx])

    # 4. 构建用于画图的 DataFrame
    df = pd.DataFrame({
        'Absolute Temporal Offset': offsets,
        'Weather Condition': weather_labels
    })

    # 5. 设置绘图风格 (学术论文常用风格)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(9, 6))

    # 6. 绘制箱线图
    # 使用调色板区分不同等级
    palette = sns.color_palette("YlGnBu", n_colors=len(level_names))
    
    ax = sns.boxplot(
        x='Weather Condition', 
        y='Absolute Temporal Offset', 
        data=df, 
        order=level_names, 
        palette=palette,
        showfliers=False, # 可以选择隐藏离群点，让箱体看起来更清晰 (True/False)
        width=0.6
    )

    # 7. 添加美化元素
    plt.title('Distribution of Alignment Offsets ($|\Delta \mathbf{P}|$) across Weather Conditions', 
              fontsize=15, pad=15, fontweight='bold')
    plt.ylabel('Magnitude of Temporal Offset', fontsize=13, fontweight='bold')
    plt.xlabel('Meteorological Categorization', fontsize=13, fontweight='bold')
    
    # 增加细微的网格线以增强可读性
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    plt.tight_layout()
    
    # 8. 保存高质量图片 (dpi=300 符合顶刊要求)
    save_path = 'offset_analysis_boxplot.pdf' # 保存为 PDF 矢量图在 LaTeX 中最清晰
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig('offset_analysis_boxplot.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ 图表已成功生成！已保存为 {save_path} 和 offset_analysis_boxplot.png")

if __name__ == '__main__':
    generate_offset_boxplot()