# Music_reprog

## Music Genre Classification with Reprogramming

<img src="https://github.com/biboamy/music-repro/blob/main/music-repro.png" width="300">

[Arxiv](https://arxiv.org/abs/2211.01317)

- Low-Resource Music Genre Classification with Advanced Neural Model Reprogramming. 
- Yun-Ning Hung, Chao-Han Huck Yang, Pin-Yu Chen, and Alexander Lerch

## How to train?
`python -u main.py --data_path ../data/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../models/reprog_resnet101Fix/ --batch_size 16 --n_epochs 200`
