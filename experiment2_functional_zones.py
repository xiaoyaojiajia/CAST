"""
实验二：不同"社会功能区"对气候扰动的异质性响应 (修订版 v2)
Heterogeneous Response of Social Functional Zones
===================================================================
修复: 归一化数据下用MSE比值代替速度降幅, 用原始测站数据计算合理降速
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from sklearn.cluster import KMeans

# ===== 1. 加载原始数据并聚类 =====
pems = np.load(r'd:\lhj\CAST\dataset\PEMSD4\PEMS03_2018_Fused_Norm.npz', allow_pickle=True)
traffic = pems['data'][:, :, 0]  # [T, 358]
weather_all = pems['mark']        # [T, 12]
T_total, N = traffic.shape

# 用原始数据聚类 (归一化后聚类等价)
hour = pd.date_range('2018-09-01', periods=T_total, freq='5min').hour.values
dow = pd.date_range('2018-09-01', periods=T_total, freq='5min').dayofweek.values

weekday = np.zeros((N, 24)); weekend = np.zeros((N, 24))
for h in range(24):
    weekday[:, h] = traffic[(hour == h) & (dow < 5)].mean(axis=0)
    weekend[:, h] = traffic[(hour == h) & (dow >= 5)].mean(axis=0)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(np.concatenate([weekday, weekend], axis=1))

# 描述聚类
for c in range(k):
    wd = weekday[clusters == c].mean(axis=0)
    we = weekend[clusters == c].mean(axis=0)
    print(f'Zone {c}: {int((clusters==c).sum())} sens, '
          f'wd_peak/mean={wd.max()/wd.mean():.2f}, wd/we_mean={wd.mean()/we.mean():.2f}')

# ===== 2. 加载预测结果 =====
tr = r'd:\lhj\CAST\test_results'
dname = 'PEMS03_Graph_CAST_PEMS_Climate_ftM_sl96_ll48_pl96_dm64_el2_df2048_eb358_weather_future_0'
pred = np.load(os.path.join(tr, dname, 'pred.npy'))
true = np.load(os.path.join(tr, dname, 'true.npy'))

offdata = np.load(os.path.join(tr, dname, 'offsets_analysis.npy'), allow_pickle=True).item()
weather_test = offdata['weathers']
rain_idx = np.where(np.argmax(weather_test, axis=1) == 4)[0]
clear_idx = np.where(np.argmax(weather_test, axis=1) == 0)[0]

# ===== 3. 用原始数据的测试集部分计算真实速度降幅 =====
# 测试集 = 后20%数据
T_test_start = int(T_total * 0.8)
test_traffic = traffic[T_test_start:]  # [T_test, 358]
test_weather = weather_all[T_test_start:]  # [T_test, 12]
is_rain_raw = test_weather[:, 11] > 0.5  # is_precip column
is_clear_raw = (np.argmax(test_weather[:, 6:11], axis=1) == 0) & ~is_rain_raw

spd_rain_full = test_traffic[is_rain_raw].mean(axis=0)  # [358]
spd_clear_full = test_traffic[is_clear_raw].mean(axis=0)

print(f'\n  原始数据: T_test={len(test_traffic)}, rainy_steps={is_rain_raw.sum()}, clear_steps={is_clear_raw.sum()}')
print(f'  is_precip 速度: {spd_rain_full.mean():.3f}, 晴天: {spd_clear_full.mean():.3f}')
print(f'  全局降速: {(spd_clear_full.mean()-spd_rain_full.mean())/abs(spd_clear_full.mean())*100:.2f}%')

# ===== 4. 分层评估 =====
print('\n' + '=' * 72)
print('  Per-Zone: Rain/Clear MSE Ratio (Normalized Data)')
print('=' * 72)
print(f'{"Zone":<12} {"Sensors":>7} {"MSE_Clear":>9} {"MSE_Rain":>9} {"Ratio":>7} {"Speed_Drop":>10}')
print('-' * 58)

for c in range(k):
    mask = clusters == c
    n_s = mask.sum()
    p_c = pred[:, :, mask]; t_c = true[:, :, mask]

    mse_c = ((p_c[clear_idx] - t_c[clear_idx])**2).mean() if len(clear_idx) > 0 else np.nan
    mse_r = ((p_c[rain_idx] - t_c[rain_idx])**2).mean() if len(rain_idx) > 0 else np.nan
    ratio = mse_r / mse_c if mse_c > 0 else np.nan

    # 真实速度降幅: 用is_precip从原始测试集计算
    sc = spd_clear_full[mask].mean()
    sr = spd_rain_full[mask].mean()
    drop = (sc - sr) / abs(sc) * 100 if abs(sc) > 1e-6 else 0

    print(f'Zone {c:<8} {n_s:>7} {mse_c:>9.4f} {mse_r:>9.4f} {ratio:>6.2f}x {drop:>9.2f}%')

mse_gc = ((pred[clear_idx] - true[clear_idx])**2).mean()
mse_gr = ((pred[rain_idx] - true[rain_idx])**2).mean()
gdrop = (spd_clear_full.mean() - spd_rain_full.mean()) / abs(spd_clear_full.mean()) * 100
print(f'{"Global":<12} {N:>7} {mse_gc:>9.4f} {mse_gr:>9.4f} {mse_gr/mse_gc:>6.2f}x {gdrop:>9.2f}%')

# ===== 5. 可视化 =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors = plt.cm.Set2(np.linspace(0, 1, k))
zone_names = [f'Zone {c}' for c in range(k)]

# (a) 日模式
ax = axes[0]
for c in range(k):
    wd = weekday[clusters == c].mean(axis=0)
    ax.plot(range(24), wd, color=colors[c], lw=2, label=f'{zone_names[c]} (n={int((clusters==c).sum())})')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Normalized Speed')
ax.set_title('(a) Diurnal Patterns by Functional Zone', fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xticks(range(0, 24, 4))

# (b) 雨天 vs 晴天 MSE
ax = axes[1]
x = np.arange(k); width = 0.35
mse_clear_vals = []; mse_rain_vals = []; ratios = []
for c in range(k):
    mask = clusters == c
    p_c = pred[:, :, mask]; t_c = true[:, :, mask]
    mse_clear_vals.append(((p_c[clear_idx] - t_c[clear_idx])**2).mean())
    mse_rain_vals.append(((p_c[rain_idx] - t_c[rain_idx])**2).mean())
    ratios.append(mse_rain_vals[-1] / mse_clear_vals[-1])
b1 = ax.bar(x - width/2, mse_clear_vals, width, label='Clear', color='gold', edgecolor='black')
b2 = ax.bar(x + width/2, mse_rain_vals, width, label='Rain', color='steelblue', edgecolor='black')
for i, (bar, r) in enumerate(zip(b2, ratios)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001, f'{r:.2f}x',
            ha='center', fontsize=9, fontweight='bold', color='darkred')
ax.set_xticks(x); ax.set_xticklabels(zone_names)
ax.set_ylabel('MSE (z-score²)'); ax.set_title('(b) Prediction Error: Clear vs Rain', fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

# (c) 雨天速度降幅
ax = axes[2]
drops = []
for c in range(k):
    sc = spd_clear_full[clusters == c].mean()
    sr = spd_rain_full[clusters == c].mean()
    drop = (sc - sr) / abs(sc) * 100 if abs(sc) > 1e-6 else 0
    drops.append(drop)
    ax.bar(c, drop, color=colors[c], edgecolor='black', width=0.5)
    ax.text(c, drop + 0.15, f'{drop:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_xticks(range(k)); ax.set_xticklabels(zone_names)
ax.set_ylabel('Speed Reduction in Rain (%)')
ax.set_title('(c) Rain-Induced Speed Drop (is_precip)', fontweight='bold')
ax.axhline(y=0, color='black', lw=0.5); ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=gdrop, color='red', ls='--', lw=1.5, alpha=0.5)
ax.text(k-0.5, gdrop+0.2, f'Global: {gdrop:.1f}%', color='red', fontsize=10, fontweight='bold')

plt.tight_layout()
os.makedirs(r'd:\lhj\CAST\figures', exist_ok=True)
plt.savefig(r'd:\lhj\CAST\figures\exp2_functional_zones.pdf', dpi=200, bbox_inches='tight')
plt.close()
print(f'\n[OK] Figure saved: figures/exp2_functional_zones.pdf')
print(f'  关键发现: 雨天MSE增幅 Zone 1={ratios[1]:.2f}x vs Zone 0={ratios[0]:.2f}x')
print(f'           速度降幅 Zone 1={drops[1]:.1f}% vs Zone 0={drops[0]:.1f}%')
