Reprogramming for music 

# Image models
## Training
#### Resnet 18 + trainable universal noise
`python main.py --data_path ../data/gtzan --model_type resnet18 --dataset gtzan --model_save_path ./../tmp/reprog_resnet18Fix_uni_noise/ --reprog_front uni_noise --fix_model True`

#### Resnet 18 + conditional noise
`python main.py --data_path ../data/gtzan --model_type resnet18 --dataset gtzan --model_save_path ./../tmp/reprog_resnet18Fix_condi/ --reprog_front condi --fix_model True`

#### Resnet 18 + conditional noise + universal noise
`python main.py --data_path ../data/gtzan --model_type resnet18 --dataset gtzan --model_save_path ./../tmp/reprog_resnet18Fix_mix/ --reprog_front mix --fix_model True`

#### Resnet 50 + trainable universal noise
`python main.py --data_path ../data/gtzan --model_type resnet50 --dataset gtzan --model_save_path ./../tmp/reprog_resnet50Fix_uni_noise/ --reprog_front uni_noise --fix_model True`

#### Resnet 50 + conditional noise
`python main.py --data_path ../data/gtzan --model_type resnet50 --dataset gtzan --model_save_path ./../tmp/reprog_resnet50Fix_condi/ --reprog_front condi --fix_model True`

#### Resnet 50 + conditional noise + universal noise
`python main.py --data_path ../data/gtzan --model_type resnet50 --dataset gtzan --model_save_path ./../tmp/reprog_resnet50Fix_mix/ --reprog_front mix --fix_model True`

#### Resnet 101 + trainable universal noise
`python main.py --data_path ../data/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../tmp/reprog_resnet101Fix_uni_noise/ --reprog_front uni_noise --fix_model True`

#### Resnet 101 + conditional noise
`python main.py --data_path ../data/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../tmp/reprog_resnet101Fix_condi/ --reprog_front condi --fix_model True`

#### Resnet 101 + conditional noise + universal noise
`python main.py --data_path ../data/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../tmp/reprog_resnet101Fix_mix/ --reprog_front mix --fix_model True`


## Evaluation
#### Resnet 18 + trainable universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet18Fix_uni_noise/best_model.pth --dataset gtzan --model_type resnet18 --reprog_front uni_noise`

#### Resnet 18 + conditional noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet18Fix_condi/best_model.pth --dataset gtzan --model_type resnet18 --reprog_front condi`

#### Resnet 18 + conditional noise + universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet18Fix_mix/best_model.pth --dataset gtzan --model_type resnet18 --reprog_front mix`

#### Resnet 50 + trainable universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet50Fix_uni_noise/best_model.pth --dataset gtzan --model_type resnet50 --reprog_front uni_noise`

#### Resnet 50 + conditional noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet50Fix_condi/best_model.pth --dataset gtzan --model_type resnet50 --reprog_front condi`

#### Resnet 50 + conditional noise + universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet50Fix_mix/best_model.pth --dataset gtzan --model_type resnet50 --reprog_front mix`

#### Resnet 101 + trainable universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet101Fix_uni_noise/best_model.pth --dataset gtzan --model_type resnet101 --reprog_front uni_noise`

#### Resnet 101 + conditional noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet101Fix_condi/best_model.pth --dataset gtzan --model_type resnet101 --reprog_front condi`

#### Resnet 101 + conditional noise + universal noise
`python -u eval.py --data_path ../data/gtzan --model_load_path ../tmp/reprog_resnet101Fix_mix/best_model.pth --dataset gtzan --model_type resnet101 --reprog_front mix`
