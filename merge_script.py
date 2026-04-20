import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 用户配置区 =================
DATASET_NAME = 'PEMS03' 
TRAFFIC_NPZ_PATH = "dataset/PEMS/PEMS04.npz"
WEATHER_CSV_PATH = "dataset/PEMSD4/dataset/San Francisco Bay Area 2018-01-01 to 2018-02-28.csv"
OUTPUT_NPZ_PATH = "dataset/PEMSD4/PEMS03_2018_Fused_Level.npz" 
# ============================================

DATASET_CONFIG = {
    'PEMS03': {'start_time': '2018-09-01 00:00:00', 'desc': 'Sacramento'},
    'PEMS04': {'start_time': '2018-01-01 00:00:00', 'desc': 'San Francisco'},
    'PEMS07': {'start_time': '2017-05-01 00:00:00', 'desc': 'Los Angeles'},
    'PEMS08': {'start_time': '2016-07-01 00:00:00', 'desc': 'San Bernardino'}
}

def process_weather_with_levels(df_raw, target_dates):
    # 1. 基础清理
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
    
    # 列名映射
    mapper = {'temperature': 'temp', 'wind_speed': 'windspeed', 'precip_accum': 'precip'}
    df_raw = df_raw.rename(columns=mapper)
    
    # 确保必要列存在
    needed_cols = ['temp', 'humidity', 'precip', 'windspeed', 'visibility', 'cloudcover']
    for c in needed_cols:
        if c not in df_raw.columns: df_raw[c] = 0

    # 2. 时间索引
    if 'datetime' in df_raw.columns:
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw = df_raw.set_index('datetime')
    else:
        df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()[~df_raw.index.duplicated(keep='first')]

    # 3. 重采样
    # 连续变量线性插值
    df_cont = df_raw[['temp', 'humidity', 'windspeed', 'visibility', 'cloudcover']].resample('5min').asfreq().interpolate(method='time')
    
    # 降水变量平摊 (用于计算等级)
    df_precip = df_raw[['precip']].resample('5min').bfill() / 12.0
    df_precip = df_precip.fillna(0)
    
    # 4. [核心] 生成降雨等级 One-Hot
    # 先计算每个时刻的等级标签
    precip_levels = df_precip['precip'].apply(get_precip_level)
    
    # 生成 One-Hot
    # 结果列如: level_No Rain, level_Drizzle, level_Light Rain...
    level_dummies = pd.get_dummies(precip_levels, prefix='rain_level')
    
    # 打印一下统计，看看是不是只有 'No Rain'
    print("\n[统计] 降雨等级分布:")
    print(precip_levels.value_counts())

    # 5. 合并
    # 我们保留原始 precip 数值(归一化后用) + 等级 One-Hot
    df_final = df_cont.join(df_precip).join(level_dummies)

    # 6. 对齐到交通时间轴
    df_aligned = df_final.reindex(target_dates)
    
    # 填充缺失值
    df_aligned = df_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_aligned, level_dummies.columns.tolist()

def run():
    print(f"=== 开始处理 (降雨分级增强版): {DATASET_NAME} ===")
    config = DATASET_CONFIG[DATASET_NAME]
    
    # 读取交通和时间
    raw_traffic = np.load(TRAFFIC_NPZ_PATH)
    traffic_data = raw_traffic['data'] if 'data' in raw_traffic else raw_traffic
    traffic_dates = pd.date_range(start=config['start_time'], periods=traffic_data.shape[0], freq='5min')

    # 读取天气
    df_weather_raw = pd.read_csv(WEATHER_CSV_PATH)
    
    # 处理
    df_weather_aligned, level_cols = process_weather_with_levels(df_weather_raw, traffic_dates)
    
    # 归一化 (只对连续数值列!)
    numeric_cols = ['temp', 'humidity', 'precip', 'windspeed', 'visibility', 'cloudcover']
    
    df_norm = df_weather_aligned.copy()
    stats_mean = df_norm[numeric_cols].mean()
    stats_std = df_norm[numeric_cols].std().replace(0, 1.0)
    df_norm[numeric_cols] = (df_norm[numeric_cols] - stats_mean) / stats_std
    
    # Level 列本身就是 0/1，不需要归一化
    
    print("   -> 数值列归一化完成。等级列保持 One-Hot。")

    # 保存
    weather_mark = df_norm.values.astype(np.float32)
    os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
    
    np.savez_compressed(
        OUTPUT_NPZ_PATH, 
        data=traffic_data, 
        mark=weather_mark, 
        columns=df_norm.columns.tolist(),
        level_cols=level_cols
    )
    
    print(f"成功保存至: {OUTPUT_NPZ_PATH}")
    print(f"Weather Feature Dim: {weather_mark.shape[1]}")
    print(f"Columns: {df_norm.columns.tolist()}")
    print("=== 请务必更新 run.py 中的 --weather_dim 参数！ ===")

if __name__ == '__main__':
    run()