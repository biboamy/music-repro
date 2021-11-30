#!/usr/bin/env bash


#python -u main.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan --model_type efficientnet_b7 --dataset gtzan --model_save_path ./../models/efficientnet_p500_m5/ --map_num 5 --pad_num 500
#python -u eval.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan/ --model_load_path ../models/compare_padding/resnet101_p500_m5/best_model.pth --dataset gtzan --model_type resnet18  --pad_num 300
#sleep 5
python -u main.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../models/reprog_resnet101_noPre/ --map_num 5 --pad_num 500 --batch_size 32
#python -u eval.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan/ --model_load_path ../models/resnet101Fix_p500_m5/best_model.pth --dataset gtzan --model_type resnet101  --pad_num 500
sleep 5
python -u main.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan --model_type resnet50 --dataset gtzan --model_save_path ./../models/reprog_resnet50_noPre/ --map_num 5 --pad_num 500 --batch_size 32
#python -u eval.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan/ --model_load_path ../models/resnet50Fix_p500_m5/best_model.pth --dataset gtzan --model_type resnet50 --pad_num 500
#sleep 5
python -u main.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan --model_type resnet18 --dataset gtzan --model_save_path ./../models/reprog_resnet18_noPre/ --map_num 5 --pad_num 500 --batch_size 32
#python -u eval.py --data_path ../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan/ --model_load_path ../models/compare_preprocess/reprog_resnet18_noNorm/best_model.pth --dataset gtzan --model_type resnet18 --pad_num 500
