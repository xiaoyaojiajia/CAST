# export CUDA_VISIBLE_DEVICES=0
# ============================================================
# PEMS08 Climate Experiment (San Bernardino)
# Dataset: PEMS08_2016_Fused_Norm.npz (170 sensors, weather_dim=12)
# ============================================================

# pred_len=12
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS08_2016_Fused_Norm.npz \
  --model_id PEMS08_2016_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
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
  --data_path PEMS08_2016_Fused_Norm.npz \
  --model_id PEMS08_2016_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
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
  --data_path PEMS08_2016_Fused_Norm.npz \
  --model_id PEMS08_2016_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
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
  --data_path PEMS08_2016_Fused_Norm.npz \
  --model_id PEMS08_2016_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30
