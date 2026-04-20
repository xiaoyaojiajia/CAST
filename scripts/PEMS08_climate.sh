# export CUDA_VISIBLE_DEVICES=0
#train
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PeMSD4/ \
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
  --weather_dim 11 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PeMSD4/ \
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
  --weather_dim 11 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PeMSD4/ \
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
  --weather_dim 11 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PeMSD4/ \
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
  --weather_dim 11 \
  --d_model 64 \
  --d_core 64 \
  --batch_size 32 \
  --learning_rate 0.003 \
  --train_epochs 30

#test
  # python -u run.py \
  # --is_training 0 \
  # --root_path ./dataset/PeMSD4/ \
  # --data_path PEMS04_2018_Fused.npz \
  # --model_id PEMS04_2018_Fused_96_96 \
  # --model CAST \
  # --data PEMS_Climate \
  # --features M \
  # --seq_len 96 \
  # --label_len 48 \
  # --pred_len 96 \
  # --enc_in 4021 \
  # --dec_in 4021 \
  # --c_out 4021 \
  # --weather_dim 7 \
  # --d_model 128 \
  # --d_core 128 \
  # --batch_size 32 \
  # --learning_rate 0.0001 \
  # --train_epochs 10 \
  # --inverse \
  # --checkpoints checkpoints\PeMSD4_Full_WithWeather_96_96_CAST_PEMS_Climate_ftM_sl96_ll48_pl96_dm128_el2_df2048_eb4021_0