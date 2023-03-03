# Music Genre Classification with Reprogramming

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<img src="https://github.com/biboamy/music-repro/blob/main/music-repro.png" width="300">

[Paper Link](https://arxiv.org/abs/2211.01317)

- Low-Resource Music Genre Classification with Advanced Neural Model Reprogramming. 
- Yun-Ning Hung, Chao-Han Huck Yang, Pin-Yu Chen, and Alexander Lerch


## Codebase
To be released, feel free to contact authors. 

- Train

```shell
python -u main.py --data_path ../data/gtzan --model_type resnet101 --dataset gtzan --model_save_path ./../models/reprog_resnet101Fix/ --batch_size 16 --n_epochs 200
```

## References

```bib
@article{hung2022low,
  title={Low-Resource Music Genre Classification with Advanced Neural Model Reprogramming},
  author={Hung, Yun-Ning and Yang, Chao-Han Huck and Chen, Pin-Yu and Lerch, Alexander},
  journal={arXiv preprint arXiv:2211.01317},
  year={2022}
}
```
