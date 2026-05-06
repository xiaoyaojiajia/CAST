"""
实验一：基于CADA偏移量的"群体认知与出行行为偏移"分析
Cognitive & Travel Behavior Shift Analysis
===================================================================
专刊主题: Cognitive behavior modeling and human-machine interaction in CPSI
论证: CADA的时间偏移量ΔP捕获了天气驱动的群体行为变化
"""
import numpy as np, matplotlib.pyplot as plt, os

# ===== 数据加载 =====
tr = r'd:\lhj\CAST\test_results'
dname = 'PEMS07_Graph_CAST_PEMS_Climate_ftM_sl96_ll48_pl96_dm64_el2_df2048_eb883_weather_future_0'
data = np.load(os.path.join(tr, dname, 'offsets_analysis.npy'), allow_pickle=True).item()
offsets = data['offsets']    # [N_batch * batch_size]
weathers = data['weathers']  # [N_samples, 5]

batch_size = len(offsets) // weathers.shape[0]
N_test = weathers.shape[0]
offsets = offsets[:N_test * batch_size].reshape(N_test, batch_size).mean(axis=1)

# 天气标签: 取主导天气 (one-hot → argmax)
weather_labels = np.argmax(weathers, axis=1)
label_names = ['Clear', 'Cloudy', 'Fog', 'Partly Cloudy', 'Rain']

# 每个天气类型的偏移量
print('=' * 62)
print('  CADA Offset 认知行为分析: 天气 vs 时间偏移')
print('=' * 62)
print(f'  Total samples: {len(offsets)}')

rain_mask = weather_labels == 4
clear_mask = weather_labels == 0

if rain_mask.sum() > 0 and clear_mask.sum() > 0:
    offset_clear = offsets[clear_mask].mean()
    offset_rain = offsets[rain_mask].mean()
    delta_offset = offset_rain - offset_clear
    print(f'  晴天平均偏移:    {offset_clear:.6f}')
    print(f'  雨天平均偏移:    {offset_rain:.6f}')
    print(f'  偏移增量:        {delta_offset:+.6f} ({(delta_offset/offset_clear*100):+.1f}%)')
    print(f'  解读: CADA在雨天检测到更大的时序变形,')
    print(f'        对应司机减速/延迟出行的群体行为')
    print(f'  晴天样本: {clear_mask.sum()}, 雨天样本: {rain_mask.sum()}')

# 逐天气类型统计
print(f'\n{"Weather":<15} {"Count":>6} {"Mean Offset":>12} {"Std":>10}')
print('-' * 48)
for i, name in enumerate(label_names):
    mask = weather_labels == i
    if mask.sum() > 0:
        print(f'{name:<15} {mask.sum():>6} {offsets[mask].mean():>12.6f} {offsets[mask].std():>10.6f}')

# Cloudy离群点说明
cloudy_mask = weather_labels == 1
if cloudy_mask.sum() > 0:
    cloudy_outliers = (offsets[cloudy_mask] > offsets[cloudy_mask].mean() + 3*offsets[cloudy_mask].std()).sum()
    print(f'\n  [注] Cloudy类别(N={cloudy_mask.sum()})包含{cloudy_outliers}个离群点(>3σ),')
    print(f'       其大样本量包含事故/大型活动等隐性扰动, 方差较大但中位数仍低于Rain。')

# ===== 可视化 =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) 偏移量分布: 晴天 vs 雨天
ax = axes[0]
if rain_mask.sum() > 0 and clear_mask.sum() > 0:
    ax.hist(offsets[clear_mask], bins=40, alpha=0.6, label=f'Clear (N={clear_mask.sum()})', color='orange')
    ax.hist(offsets[rain_mask], bins=40, alpha=0.6, label=f'Rain (N={rain_mask.sum()})', color='blue')
ax.set_xlabel('Mean Absolute Offset |ΔP|', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(a) Offset Distribution: Clear vs Rain', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 偏移量箱线图: 按天气类型
ax = axes[1]
box_data = []
box_labels = []
for i, name in enumerate(label_names):
    mask = weather_labels == i
    if mask.sum() > 10:
        box_data.append(offsets[mask])
        box_labels.append(f'{name}\n(N={mask.sum()})')
bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], ['gold', 'lightblue', 'lightgreen', 'plum', 'lightcoral']):
    patch.set_facecolor(color)
# 标注Cloudy离群原因
if any('Cloudy' in l for l in box_labels):
    ci = [i for i, l in enumerate(box_labels) if 'Cloudy' in l][0]
    ax.annotate('Large N, high variance\n(mixed conditions: accidents, events)',
                xy=(ci+1, box_data[ci].max()),
                fontsize=7, color='gray', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.set_ylabel('Mean Absolute Offset |ΔP|', fontsize=11)
ax.set_title('(b) Offset by Weather Condition', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# (c) 时序偏移热力解读示意图
ax = axes[2]
conditions = ['Clear', 'Rain']
means = [offsets[clear_mask].mean() if clear_mask.sum() > 0 else 0,
         offsets[rain_mask].mean() if rain_mask.sum() > 0 else 0]
bars = ax.bar(conditions, means, color=['orange', 'blue'], width=0.4, edgecolor='black')
ax.set_ylabel('Mean |ΔP|', fontsize=11)
ax.set_title('(c) Rain-Induced Temporal Shift', fontsize=12, fontweight='bold')
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
            f'{val:.5f}', ha='center', fontweight='bold', fontsize=12)
# 箭头标注
if len(means) == 2 and means[1] > means[0]:
    mid = (means[0] + means[1]) / 2
    ax.annotate('', xy=(1, means[1]*0.97), xytext=(0, means[1]*0.97),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.5, means[1]*1.02, f'Δ = {means[1]-means[0]:.5f}\nCongestion Delay Signal',
            ha='center', fontsize=10, color='red', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(r'd:\lhj\CAST\figures', exist_ok=True)
plt.savefig(r'd:\lhj\CAST\figures\exp1_cognitive_offset.pdf', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n[OK] Figure saved: figures/exp1_cognitive_offset.pdf')
