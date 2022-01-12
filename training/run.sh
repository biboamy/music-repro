#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=1 python3 main.py --data_path /media/sdc/amy/data/FMA --model_type speechatt --dataset FMA --model_save_path ./../FMA/reprog_speechattFix_condi_spec/ --reprog_front condi --fix_model True --map_num 2 

#CUDA_VISIBLE_DEVICES=1 python3 main.py --data_path /media/sdc/amy/data/FMA --model_type speechatt --dataset FMA --model_save_path ./../FMA/reprog_speechattFix_uni_noise_spec/ --reprog_front uni_noise --fix_model True --map_num 2 

#CUDA_VISIBLE_DEVICES=1 python3 main.py --data_path /media/sdc/amy/data/FMA --model_type resnet50 --dataset FMA --model_save_path ./../FMA/padding/reprog_resnet50Fix_mix_200/ --reprog_front mix --fix_model True --pad_num 200

#CUDA_VISIBLE_DEVICES=1 python3 main.py --data_path /media/sdc/amy/data/FMA --model_type resnet50 --dataset FMA --model_save_path ./../FMA/padding/reprog_resnet50Fix_mix_100/ --reprog_front mix --fix_model True --pad_num 100

CUDA_VISIBLE_DEVICES=1 python3 -W ignore -u eval.py --data_path /media/sdc/amy/data/FMA --model_load_path ../FMA/reprog_speechattFix_condi_spec/best_model.pth --dataset FMA --model_type speechatt --reprog_front condi --map_num 2 

#CUDA_VISIBLE_DEVICES=1 python3 -W ignore -u eval.py --data_path /media/sdc/amy/data/FMA --model_load_path ../FMA/reprog_speechattFix_uni_noise_spec/best_model.pth --dataset FMA --model_type speechatt --reprog_front uni_noise --map_num 2 

#CUDA_VISIBLE_DEVICES=1 python3 -W ignore -u eval.py --data_path /media/sdc/amy/data/FMA --model_load_path ../FMA/padding/reprog_resnet50Fix_mix_200/best_model.pth --dataset FMA --model_type resnet50 --reprog_front mix --pad_num 200

#CUDA_VISIBLE_DEVICES=1 python3 -W ignore -u eval.py --data_path /media/sdc/amy/data/FMA --model_load_path ../FMA/padding/reprog_resnet50Fix_mix_100/best_model.pth --dataset FMA --model_type resnet50 --reprog_front mix --pad_num 100

