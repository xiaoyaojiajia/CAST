export CUDA_VISIBLE_DEVICES=0

#train
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id CAST_PEMS08_96_12 \
  --model CAST \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 12 \
  --e_layers 4 \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --itr 1 \
  --weather_dim 1 \
  --use_norm 1 \


  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id CAST_PEMS08_96_24 \
  --model CAST \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 4 \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --itr 1 \
  --use_norm 1 \
  --weather_dim 1 

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id CAST_PEMS08_96_48 \
  --model CAST \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --e_layers 4 \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --itr 1 \
  --use_norm 1 \
  --weather_dim 1 

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id CAST_PEMS08_96_96 \
  --model CAST \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_core 512 \
  --d_ff 1024 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --use_norm 1 \
  --itr 1 \
  --weather_dim 1 

#test
#     python -u run.py \
#   --is_training 0 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS08.npz \
#   --model_id CAST_PEMS08_96_96 \
#   --model CAST \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 4 \
#   --d_model 512 \
#   --d_core 512 \
#   --d_ff 1024 \
#   --enc_in 170 \
#   --dec_in 170 \
#   --c_out 170 \
#   --batch_size 32 \
#   --learning_rate 0.0003 \
#   --itr 1 \
#   --weather_dim 1 
#   --checkpoints checkpoints\CAST_PEMS08_96_96