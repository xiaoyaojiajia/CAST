# export CUDA_VISIBLE_DEVICES=0
#train
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PeMSD4/ \
#   --data_path PEMS04_2018_Fused.npz \
#   --model_id PEMS04_2018_WithWeather \
#   --model CAST \
#   --data PEMS_Climate \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 12 \
#   --enc_in 307 \
#   --dec_in 307 \
#   --c_out 307 \
#   --weather_dim 7 \
#   --d_model 64 \
#   --d_core 64 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --train_epochs 30

#   python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PeMSD4/ \
#   --data_path PEMS04_2018_Fused.npz \
#   --model_id PEMS04_2018_WithWeather \
#   --model CAST \
#   --data PEMS_Climate \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 24 \
#   --enc_in 307 \
#   --dec_in 307 \
#   --c_out 307 \
#   --weather_dim 7 \
#   --d_model 64 \
#   --d_core 64 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --train_epochs 30

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PeMSD4/ \
#   --data_path PEMS04_2018_Fused.npz \
#   --model_id PEMS04_2018_WithWeather \
#   --model CAST \
#   --data PEMS_Climate \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 48 \
#   --enc_in 307 \
#   --dec_in 307 \
#   --c_out 307 \
#   --weather_dim 7 \
#   --d_model 64 \
#   --d_core 64 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --train_epochs 30

#   python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PeMSD4/ \
#   --data_path PEMS04_2018_Fused.npz \
#   --model_id PEMS04_2018_WithWeather \
#   --model CAST \
#   --data PEMS_Climate \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --enc_in 307 \
#   --dec_in 307 \
#   --c_out 307 \
#   --weather_dim 7 \
#   --d_model 64 \
#   --d_core 64 \
#   --batch_size 32 \
#   --learning_rate 0.001 \
#   --train_epochs 30

#test
python -u run.py \
  --is_training 0 \
  --root_path ./dataset/PeMSD4/ \
  --data_path PEMS04_2018_Fused.npz \
  --model_id PEMS04_2018_WithWeather \
  --model CAST \
  --data PEMS_Climate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --weather_dim 7 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --inverse \
  --checkpoints ./checkpoints/
  