# coding: utf-8
import os
import numpy as np
from torch.utils import data
import random
import soundfile as sf
import librosa


class GTZAN(data.Dataset):
    def __init__(self, root, split, input_length=None):
        split = split.lower()
        self.mappeing = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9,
        }
        self.files = [
            f
            for f in open(f"{root}/{split}_filtered.txt", "r").readlines()
            if "jazz.00054" not in f
        ]
        self.class_num = 10
        self.split = split
        self.seg_length = input_length
        self.root = root

    def __len__(self):
        if self.split == "train":
            return 1000
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if self.split == "train":
            idx = random.randint(0, len(self.files) - 1)
        file = self.files[idx].strip()
        frame = sf.info(os.path.join(self.root, file)).frames
        label = np.zeros(self.class_num)
        label[self.mappeing[file.split("/")[0]]] = 1
        if self.split == "train":
            audio, sr = librosa.load(os.path.join(self.root, file), sr=16000)
            start = random.randint(0, len(audio) - self.seg_length - 16000)
            audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            return audio, label.astype("float32")
        else:
            audio, sr = librosa.load(os.path.join(self.root, file), sr=16000)
            audio = audio.astype("float32")
            n_chunk = len(audio) // self.seg_length
            audio_chunks = np.split(audio[: int(n_chunk * self.seg_length)], n_chunk)
            audio_chunks.append(audio[-int(self.seg_length) :])
            audio = np.array(audio_chunks)

            return audio, label.astype("float32")


def get_audio_loader(
    root,
    batch_size,
    split="TRAIN",
    num_workers=0,
    input_length=None
):
    data_loader = data.DataLoader(
        dataset=GTZAN(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader
