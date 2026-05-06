import subprocess, sys
cmd = [
    r'D:\anaconda3\envs\basicTS\python.exe', '-u', 'run.py',
    '--is_training', '1', '--root_path', './dataset/PEMSD4/',
    '--data_path', 'PEMS08_2016_Fused_Norm.npz',
    '--model_id', 'PEMS08_2016_WithWeather', '--model', 'CAST',
    '--data', 'PEMS_Climate', '--features', 'M',
    '--seq_len', '96', '--label_len', '48', '--pred_len', '12',
    '--enc_in', '170', '--dec_in', '170', '--c_out', '170',
    '--weather_dim', '12', '--use_future_weather', '1', '--has_weather', '1',
    '--d_model', '64', '--d_core', '64', '--batch_size', '32',
    '--learning_rate', '0.003', '--train_epochs', '10', '--patience', '5',
    '--num_workers', '0',
]
print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, cwd=r'd:\lhj\CAST', capture_output=True, text=True, timeout=600)
print('STDOUT:', result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print('STDERR:', result.stderr[-500:] if result.stderr else 'NONE')
print('EXIT:', result.returncode)
