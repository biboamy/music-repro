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

3. Install Dependencies
```
pip3 install -r requirement.txt
```

4. Pull pre-trained models
```
git lfs fetch --all
```


5. run experiment (skip to "7" for inference only)

```
python3 main.py --reprog_front uni_noise

python3 main.py --reprog_front condi

python3 main.py --reprog_front skip
```

6. Visit "demo.ipynb" for inference only demo


## AST Reference

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

## Citing Music Reprogramming

```bib
@inproceedings{hung2023low,
  title={Low-Resource Music Genre Classification with Cross-Modal Neural Model Reprogramming},
  author={Hung, Yun-Ning and Yang, Chao-Han Huck and Chen, Pin-Yu and Lerch, Alexander},
  booktitle={Proc. of ICASSP 2023},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Bug Fix

If you encounter the following errors "**batch response: This repository is over its data quota. Account responsible for LFS...**", Please download the model from here [Google Drive](https://drive.google.com/file/d/1XDcvNN5kYdd7J7bOHDH_TwTCsed1RGj5/view?usp=sharing)

