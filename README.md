# Music Genre Classification with Reprogramming

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

<img src="https://github.com/biboamy/music-repro/blob/main/music-repro.png" width="300">

[Paper Link](https://arxiv.org/abs/2211.01317)

- Low-Resource Music Genre Classification with Advanced Neural Model Reprogramming. 
- Yun-Ning Hung, Chao-Han Huck Yang, Pin-Yu Chen, and Alexander Lerch


## Codebase

1. Download GTZAN dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

2. unzip and move file:

```
unzip archive.zip
mv Data/genres_original/* music-repro/data/
```

3. download pre-trained model:  https://www.dropbox.com/s/ca0b1v2nlxzyeb4/audioset_10_10_0.4593.pth?dl=1
```
mv audioset_10_10_0.4593.pth training/models/
```

4. Install Dependencies
```
pip3 install -r requirement.txt
```


4. run experiment

```
python3 main.py --reprog_front uni_noise

python3 main.py --reprog_front condi

python3 main.py --reprog_front skip
```

## Reference

```bib
@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}
```

The ast code used in this repo comes from the [original repo](https://github.com/YuanGongND/ast)

## Citing

```bib
@article{hung2022low,
  title={Low-Resource Music Genre Classification with Advanced Neural Model Reprogramming},
  author={Hung, Yun-Ning and Yang, Chao-Han Huck and Chen, Pin-Yu and Lerch, Alexander},
  journal={arXiv preprint arXiv:2211.01317},
  year={2022}
}
```
