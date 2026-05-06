import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

DATASET_CONFIG = {
    'PEMS03': {
        'traffic_npz': 'dataset/PEMS/PEMS03.npz',
        'weather_csv': 'dataset/PEMSD4/Sacramento 2018-09-01 to 2018-11-30.csv',
        'start_time': '2018-09-01 00:00:00',
        'out_npz': 'dataset/PEMSD4/PEMS03_2018_Fused_Norm.npz',
        'enc_in': 358,
    },
    'PEMS04': {
        'traffic_npz': 'dataset/PEMS/PEMS04.npz',
        'weather_csv': 'dataset/PEMSD4/San Francisco Bay Area 2018-01-01 to 2018-02-28.csv',
        'start_time': '2018-01-01 00:00:00',
        'out_npz': 'dataset/PEMSD4/PEMS04_2018_Fused_Norm.npz',
        'enc_in': 307,
    },
    'PEMS07': {
        'traffic_npz': 'dataset/PEMS/PEMS07.npz',
        'weather_csv': 'dataset/PEMSD4/Los Angeles 2017-05-01 to 2017-08-31.csv',
        'start_time': '2017-05-01 00:00:00',
        'out_npz': 'dataset/PEMSD4/PEMS07_2017_Fused_Norm.npz',
        'enc_in': 883,
    },
    'PEMS08': {
        'traffic_npz': 'dataset/PEMS/PEMS08.npz',
        'weather_csv': 'dataset/PEMSD4/San Bernardino 2016-07-01 to 2016-08-31.csv',
        'start_time': '2016-07-01 00:00:00',
        'out_npz': 'dataset/PEMSD4/PEMS08_2016_Fused_Norm.npz',
        'enc_in': 170,
    },
}

ICON_CATEGORIES = ['clear', 'cloudy', 'fog', 'partly-cloudy', 'rain']


def process_weather(df_raw, target_dates):
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
    df_raw = df_raw.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='first')
    df_raw = df_raw.set_index('datetime')

    needed = ['temp', 'humidity', 'precip', 'windspeed', 'visibility', 'cloudcover']
    for c in needed:
        if c not in df_raw.columns:
            df_raw[c] = 0.0

    for c in needed:
        df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0.0)

    if 'icon' in df_raw.columns:
        df_raw['icon'] = df_raw['icon'].str.strip().str.lower().fillna('clear')
    else:
        df_raw['icon'] = 'clear'

    df_cont = df_raw[needed].resample('5min').asfreq().interpolate(method='time')
    df_precip = df_raw[['precip']].resample('5min').bfill() / 12.0
    df_precip = df_precip.fillna(0.0)

    if 'icon' in df_raw.columns:
        icon_5min = df_raw[['icon']].resample('5min').ffill()
    else:
        icon_5min = pd.DataFrame(index=df_cont.index, data={'icon': 'clear'})

    for ic in ICON_CATEGORIES:
        df_cont[f'icon_{ic}'] = (icon_5min['icon'] == ic).astype(np.float32)

    df_cont['is_precip'] = (df_precip['precip'] > 0.001).astype(np.float32)
    df_cont['precip'] = df_precip['precip']

    df_aligned = df_cont.reindex(target_dates)
    df_aligned = df_aligned.ffill().bfill().fillna(0.0)

    return df_aligned


def run_dataset(name, cfg):
    print(f'\n{"="*60}')
    print(f'Processing {name}: {cfg["weather_csv"]}')
    print(f'{"="*60}')

    raw_traffic = np.load(cfg['traffic_npz'], allow_pickle=True)
    traffic_data = raw_traffic['data']
    print(f'  Traffic shape: {traffic_data.shape}')

    T = traffic_data.shape[0]
    traffic_dates = pd.date_range(start=cfg['start_time'], periods=T, freq='5min')
    print(f'  Time range: {traffic_dates[0]} to {traffic_dates[-1]}, steps={T}')

    df_weather_raw = pd.read_csv(cfg['weather_csv'])
    print(f'  Weather rows: {len(df_weather_raw)}')

    df_aligned = process_weather(df_weather_raw, traffic_dates)

    numeric_cols = ['temp', 'humidity', 'precip', 'windspeed', 'visibility', 'cloudcover']
    icon_cols = [f'icon_{ic}' for ic in ICON_CATEGORIES]

    stats_mean = df_aligned[numeric_cols].mean()
    stats_std = df_aligned[numeric_cols].std().replace(0, 1.0)
    df_aligned[numeric_cols] = (df_aligned[numeric_cols] - stats_mean) / stats_std

    all_cols = numeric_cols + icon_cols + ['is_precip']
    weather_mark = df_aligned[all_cols].values.astype(np.float32)

    print(f'  Weather dim: {weather_mark.shape[1]}')
    print(f'  Columns: {all_cols}')
    print(f'  icon_cols: {icon_cols}')

    os.makedirs(os.path.dirname(cfg['out_npz']), exist_ok=True)
    np.savez_compressed(
        cfg['out_npz'],
        data=traffic_data,
        mark=weather_mark,
        columns=np.array(all_cols),
        icon_cols=np.array(icon_cols),
    )
    print(f'  Saved: {cfg["out_npz"]}')
    print(f'  data shape: {traffic_data.shape}, mark shape: {weather_mark.shape}')


def main():
    for name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        run_dataset(name, DATASET_CONFIG[name])

    print('\n' + '='*60)
    print('SUMMARY')
    for name in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        cfg = DATASET_CONFIG[name]
        data = np.load(cfg['out_npz'], allow_pickle=True)
        print(f'  {name}: data={data["data"].shape}, mark={data["mark"].shape}, '
              f'weather_dim={data["mark"].shape[1]}, enc_in={cfg["enc_in"]}')


if __name__ == '__main__':
    main()
