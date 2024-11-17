export CUDA_VISIBLE_DEVICES=0

# Training phase
python main.py --anormly_ratio 1 --num_epochs 3 --batch_size 128 --mode train --dataset GSF --data_path dataset/GSF --input_c 17 --output_c 17

# Testing phase
python main.py --anormly_ratio 1 --num_epochs 10 --batch_size 128 --mode test --dataset GSF --data_path dataset/GSF --input_c 17 --output_c 17 --pretrained_model 20
