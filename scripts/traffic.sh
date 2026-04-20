export CUDA_VISIBLE_DEVICES=0

# 1. 预测长度 96
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id CAST_Traffic_96_96 \
  --model CAST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --lradj type1 \
  --train_epochs 50 \
  --patience 20 \
  --itr 1 \
  --num_workers 0

# 2. 预测长度 192
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id CAST_Traffic_96_192 \
  --model CAST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --lradj type1 \
  --train_epochs 50 \
  --patience 20 \
  --itr 1 \
  --num_workers 0

# 3. 预测长度 336
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id CAST_Traffic_96_336 \
  --model CAST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --lradj type1 \
  --train_epochs 50 \
  --patience 20 \
  --itr 1 \
  --num_workers 0

# 4. 预测长度 720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id CAST_Traffic_96_720 \
  --model CAST \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_model 512 \
  --d_core 128 \
  --d_ff 512 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 16 \
  --learning_rate 0.0003 \
  --lradj type1 \
  --train_epochs 50 \
  --patience 20 \
  --itr 1 \
  --num_workers 0