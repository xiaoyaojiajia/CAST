import subprocess, os, sys
BASE = r'D:\anaconda3\envs\basicTS\python.exe'
WD = r'd:\lhj\CAST'
logfile = os.path.join(WD, '_run_log.txt')

exp = sys.argv[1]  # 'p08_24', 'p08_48', etc.

configs = {
    'p08_24': '--data_path PEMS08_2016_Fused_Norm.npz --model_id PEMS08_2016_WithWeather --pred_len 24 --enc_in 170 --dec_in 170 --c_out 170 --batch_size 32 --learning_rate 0.003',
    'p08_48': '--data_path PEMS08_2016_Fused_Norm.npz --model_id PEMS08_2016_WithWeather --pred_len 48 --enc_in 170 --dec_in 170 --c_out 170 --batch_size 32 --learning_rate 0.003',
    'p08_96': '--data_path PEMS08_2016_Fused_Norm.npz --model_id PEMS08_2016_WithWeather --pred_len 96 --enc_in 170 --dec_in 170 --c_out 170 --batch_size 32 --learning_rate 0.003',
    'p07_12': '--data_path PEMS07_2017_Fused_Norm.npz --model_id PEMS07_2017_WithWeather --pred_len 12 --enc_in 883 --dec_in 883 --c_out 883 --batch_size 16 --learning_rate 0.003',
    'p07_24': '--data_path PEMS07_2017_Fused_Norm.npz --model_id PEMS07_2017_WithWeather --pred_len 24 --enc_in 883 --dec_in 883 --c_out 883 --batch_size 16 --learning_rate 0.003',
    'p07_48': '--data_path PEMS07_2017_Fused_Norm.npz --model_id PEMS07_2017_WithWeather --pred_len 48 --enc_in 883 --dec_in 883 --c_out 883 --batch_size 16 --learning_rate 0.003',
    'p07_96': '--data_path PEMS07_2017_Fused_Norm.npz --model_id PEMS07_2017_WithWeather --pred_len 96 --enc_in 883 --dec_in 883 --c_out 883 --batch_size 16 --learning_rate 0.003',
}

cfg = configs[exp]
cmd = f'{BASE} -u run.py --is_training 1 --root_path ./dataset/PEMSD4/ {cfg} --model CAST --data PEMS_Climate --features M --seq_len 96 --label_len 48 --weather_dim 12 --use_future_weather 1 --has_weather 1 --d_model 64 --d_core 64 --train_epochs 30 --patience 10 --num_workers 0'

with open(logfile, 'a', encoding='utf-8') as f:
    f.write(f'\n=== {exp} START ===\n{cmd}\n')
    f.flush()
    result = subprocess.run(cmd, cwd=WD, stdout=f, stderr=subprocess.STDOUT, timeout=3600, shell=True)
    f.write(f'\n=== {exp} EXIT: {result.returncode} ===\n')

print(f'{exp} DONE rc={result.returncode}')
