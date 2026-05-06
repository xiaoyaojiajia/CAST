import subprocess, os

BASE = r'D:\anaconda3\envs\basicTS\python.exe'
WD = r'd:\lhj\CAST'
LOG_DIR = r'd:\lhj\CAST\_logs'
os.makedirs(LOG_DIR, exist_ok=True)

experiments = [
    # PEMS08: 24, 48, 96
    ('PEMS08_24', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS08_2016_Fused_Norm.npz', '--model_id', 'PEMS08_2016_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '24', '--enc_in', '170', '--dec_in', '170', '--c_out', '170', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '32', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    ('PEMS08_48', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS08_2016_Fused_Norm.npz', '--model_id', 'PEMS08_2016_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '48', '--enc_in', '170', '--dec_in', '170', '--c_out', '170', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '32', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    ('PEMS08_96', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS08_2016_Fused_Norm.npz', '--model_id', 'PEMS08_2016_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '96', '--enc_in', '170', '--dec_in', '170', '--c_out', '170', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '32', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    # PEMS07: 12, 24, 48, 96 (batch_size=16, 883 sensors)
    ('PEMS07_12', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS07_2017_Fused_Norm.npz', '--model_id', 'PEMS07_2017_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '12', '--enc_in', '883', '--dec_in', '883', '--c_out', '883', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '16', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    ('PEMS07_24', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS07_2017_Fused_Norm.npz', '--model_id', 'PEMS07_2017_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '24', '--enc_in', '883', '--dec_in', '883', '--c_out', '883', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '16', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    ('PEMS07_48', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS07_2017_Fused_Norm.npz', '--model_id', 'PEMS07_2017_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '48', '--enc_in', '883', '--dec_in', '883', '--c_out', '883', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '16', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
    ('PEMS07_96', ['run.py', '--is_training', '1', '--root_path', './dataset/PEMSD4/', '--data_path', 'PEMS07_2017_Fused_Norm.npz', '--model_id', 'PEMS07_2017_WithWeather', '--model', 'CAST', '--data', 'PEMS_Climate', '--features', 'M', '--seq_len', '96', '--label_len', '48', '--pred_len', '96', '--enc_in', '883', '--dec_in', '883', '--c_out', '883', '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1', '--d_model', '64', '--d_core', '64', '--batch_size', '16', '--learning_rate', '0.003', '--train_epochs', '30', '--patience', '10', '--num_workers', '0']),
]

for name, args in experiments:
    log_path = os.path.join(LOG_DIR, f'{name}.log')
    print(f'\n{"="*60}')
    print(f'Starting: {name} -> {log_path}')
    print(f'{"="*60}')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        result = subprocess.run([BASE, '-u'] + args, cwd=WD, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    
    # Extract MSE from log
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    for line in content.split('\n'):
        if 'mse:' in line.lower():
            print(f'  RESULT: {line.strip()}')
    print(f'  Exit code: {result.returncode}')
