# export CUDA_VISIBLE_DEVICES=0
# ============================================================
# PEMS07 Graph Experiment (Los Angeles, 883 sensors)
# 密集图扩散: softmax(relu(E@E^T)) 捕获传感器空间关系
# ============================================================

# pred_len=12
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS07_2017_Fused_Norm.npz \
  --model_id PEMS07_Graph \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --train_epochs 30 \
  --patience 10 \
  --num_workers 0

# pred_len=24
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS07_2017_Fused_Norm.npz \
  --model_id PEMS07_Graph \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --train_epochs 30 \
  --patience 10 \
  --num_workers 0

# pred_len=48
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS07_2017_Fused_Norm.npz \
  --model_id PEMS07_Graph \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --train_epochs 30 \
  --patience 10 \
  --num_workers 0

# pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMSD4/ \
  --data_path PEMS07_2017_Fused_Norm.npz \
  --model_id PEMS07_Graph \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --weather_dim 12 \
  --use_future_weather 1 \
  --has_weather 1 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 16 \
  --learning_rate 0.003 \
  --train_epochs 30 \
  --patience 10 \
  --num_workers 0
