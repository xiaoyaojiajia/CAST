# export CUDA_VISIBLE_DEVICES=0
# ============================================================
# PEMS03 Climate Experiment (Sacramento)
# Dataset: PEMS03_2018_Fused_Norm.npz (358 sensors, weather_dim=12)
# ============================================================

# pred_len=12
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS03_2018_Fused_Norm.npz \
  --model_id PEMS03_2018_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

# pred_len=24
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS03_2018_Fused_Norm.npz \
  --model_id PEMS03_2018_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

# pred_len=48
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS03_2018_Fused_Norm.npz \
  --model_id PEMS03_2018_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

# pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS03_2018_Fused_Norm.npz \
  --model_id PEMS03_2018_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30
