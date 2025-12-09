export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id CAST_PEMS04_96_96 \
  --model CAST \
  --data PEMS \
  --features M \
  --seq_len 12 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 2 \
  --d_model 128 \
  --d_core 128 \
  --d_ff 256 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --itr 1