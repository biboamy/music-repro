import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio


class AttRNNSpeechModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(AttRNNSpeechModel, self).__init__()
        self.decibel = True
        self.mel_extr = torchaudio.transforms.MelSpectrogram(
            n_fft=1024, n_mels=80, f_min=40, f_max=8000, power=1
        )

        if self.decibel:
            self.mag2db = torchaudio.transforms.AmplitudeToDB(80)
        """
        kernal = 130
        self.skip_conv = nn.Sequential(
            nn.Conv1d(128, kernal, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(kernal),
            nn.Conv1d(kernal, kernal, 3, 3),
            nn.ReLU(),
            nn.BatchNorm1d(kernal),
            nn.Conv1d(kernal, kernal, 3, 3),
            nn.ReLU(),
            nn.BatchNorm1d(kernal),
            nn.Conv1d(kernal, kernal, 3, 3)
        )
        self.skip_linear = nn.Linear(kernal, 128*2)
        """
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=(5, 1), padding="same"),
            torch.nn.BatchNorm2d(10, eps=0.001),
            torch.nn.Conv2d(10, 1, kernel_size=(5, 1), padding="same"),
            torch.nn.BatchNorm2d(1, eps=0.001),
        )

        self.lstm_block = torch.nn.LSTM(
            input_size=80,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.query_transform = torch.nn.Linear(128, 128)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.Linear(64, 32), torch.nn.Linear(32, 35)
        )
        # torch.nn.Softmax(dim=1))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                torch.nn.init.orthogonal_(param.data)
            elif "weight" in name:
                if param.data.ndim == 1:
                    param.data = param.data.unsqueeze(-1)
                    torch.nn.init.xavier_uniform_(param.data)
                    param.data = param.data.squeeze(-1)
                else:
                    torch.nn.init.xavier_uniform_(param.data)
            elif "bias" in name:
                torch.nn.init.constant_(param.data, 0)

    def normalize2d(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        return (x - mean) / (std + 1e-10)

    def transform(self, features, delta=None):
        # vgg = self.skip_linear(self.skip_conv(features.squeeze(1).permute(0, 2, 1)).squeeze(-1))
        # gammas = vgg[:, :128]
        # betas = vgg[:, 128:]

        features = self.conv_block(features).squeeze(1)
        # vgg = self.skip_linear(self.skip_conv(features.permute(0, 2, 1)).squeeze(-1))
        # gammas = vgg[:, :128]
        # betas = vgg[:, 128:]
        hidden_seq, _ = self.lstm_block(features)
        # vgg = self.skip_linear(self.skip_conv(hidden_seq.permute(0, 2, 1)).squeeze(-1))
        # gammas = vgg[:, :128]
        # betas = vgg[:, 128:]
        query = self.query_transform(hidden_seq.mean(dim=1)).view(
            hidden_seq.shape[0], 1, hidden_seq.shape[-1]
        )
        att_scores = torch.bmm(query, hidden_seq.permute(0, 2, 1))
        predicted = torch.bmm(att_scores, hidden_seq).squeeze(1)
        # predicted = (1 + gammas) * predicted + betas
        # predicted += vgg

        predicted = self.head(predicted)
        return predicted

    def forward(self, features, target, **kwargs):
        predicted = self.transform(features)
        return torch.nn.functional.nll_loss(
            torch.log(predicted), target, reduction="sum"
        )


class V2SReprogModel(torch.nn.Module):
    def __init__(self, map_num=5, n_class=16, reprog_front=None):
        super(V2SReprogModel, self).__init__()

        self.cls_model = self.load()

        for name, param in self.cls_model.named_parameters():
            if "skip" not in name:
                param.requires_grad = False

        if reprog_front == "uni_noise":
            self.delta = torch.nn.Parameter(
                torch.Tensor(1, 1, 16000), requires_grad=True
            )
            torch.nn.init.xavier_uniform_(self.delta)
        elif reprog_front == "condi":
            n_channel = 140  # 50

            self.linear = nn.Linear(n_channel, 80)
            self.conv = nn.Sequential(
                nn.Conv1d(80, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm1d(n_channel),
                nn.Conv1d(n_channel, n_channel, 3, 1, 1),
            )

        elif reprog_front == "mix":
            self.delta = torch.nn.Parameter(
                torch.Tensor(1, pad_num * 3 * 157), requires_grad=True
            )
            self.linear_emb = nn.Linear(152 * 3, pad_num * 3)
            self.linear_com = nn.Sequential(
                nn.Conv1d(pad_num * 3, pad_num * 3, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(pad_num * 3, pad_num * 3, 3, 1, 1),
            )
            torch.nn.init.xavier_uniform_(self.delta)

        # self.lmbd = args.lmbd
        self.map_num = map_num
        self.class_num = n_class
        self.reprog_front = reprog_front

    def load(self):
        ckpt = torch.load("AttRNNSpeechModel.pth", map_location="cpu")
        args = ckpt["Args"]

        model = eval(f"{args.model}")(args)

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt["Model"].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, wav):
        n_batch = wav.shape[0]
        wav = wav.reshape(n_batch, -1, 16000).reshape(-1, 16000).unsqueeze(1)

        if self.reprog_front == "uni_noise" or self.reprog_front == "mix":
            wav = wav + self.delta

        features = self.cls_model.mel_extr(wav).permute(0, 1, 3, 2)
        features = self.cls_model.mag2db(features)
        if self.reprog_front == "condi" or self.reprog_front == "mix":
            features = self.linear(
                self.conv(features.squeeze(1).permute(0, 2, 1)).permute(0, 2, 1)
            ).unsqueeze(1)

        predicted = self.cls_model.transform(features)[
            :, : self.class_num * self.map_num
        ]

        predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
        predicted = predicted.reshape(n_batch, -1, self.class_num).mean(1)
        return predicted
