export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id CAST_traffic_192_96 \
  --model CAST \
  --data custom \
  --features M \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 128 \
  --d_core 128 \
  --d_ff 256 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1