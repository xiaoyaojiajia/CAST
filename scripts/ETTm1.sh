export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id CAST_ETTm1_96_96 \
  --model CAST \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 128 \
  --d_core 128 \
  --d_ff 256 \
  --enc_in 7 \
  --c_out 7 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1
