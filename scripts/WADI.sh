export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1 --num_epochs 3   --batch_size 128  --mode train --dataset WADI  --data_path dataset/WADI --input_c 123    --output_c 123
python main.py --anormly_ratio 1  --num_epochs 10      --batch_size 128     --mode test    --dataset WADI   --data_path dataset/WADI  --input_c 123    --output_c 123  --pretrained_model 20




