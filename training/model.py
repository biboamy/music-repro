# coding: utf-8
import torch
import torch.nn as nn
import torchaudio

from modules import HarmonicSTFT


class Conv3_2d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, kernal=3):
        super(Conv3_2d, self).__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernal,
            padding=1
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Conv3_2d_resmp(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        pooling=2,
        kernal=3,
        padding=1
    ):
        super(Conv3_2d_resmp, self).__init__()
        self.conv_1 = nn.Conv2d(
            input_channels, output_channels, kernal, padding=padding
        )
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(
            output_channels, output_channels, kernal, padding=padding
        )
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class CNNModel(torch.nn.Module):
    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=10,
        n_harmonic=3,
        semitone_scale=2,
        learn_bw="only_Q",
        model_type="CNN16k",
    ):
        super(CNNModel, self).__init__()
        self.hcqt = HarmonicSTFT(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_harmonic=n_harmonic,
            semitone_scale=semitone_scale,
            learn_bw=learn_bw,
        )

        if model_type == "CNN235.5k":
            conv_channels = 54
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 2, kernal=6),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
            )
            self.classifier = nn.Linear(4374, n_class)
            # 4698 58
            # 4374 54

    def forward(self, features, **kwargs):
        x = self.hcqt(features)
        x = self.conv(x).reshape(len(x), -1)
        # print(x.shape)
        x = self.classifier(x)

        return x


class AttRNNSpeechModel(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super(AttRNNSpeechModel, self).__init__()
        self.decibel = True
        self.mel_extr = torchaudio.transforms.MelSpectrogram(
            n_fft=1024, n_mels=80, f_min=40, f_max=8000, power=1
        )

        if self.decibel:
            self.mag2db = torchaudio.transforms.AmplitudeToDB(80)

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
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 32),
            torch.nn.Linear(32, 35)
        )

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
        features = self.conv_block(features).squeeze(1)
        hidden_seq, _ = self.lstm_block(features)
        query = self.query_transform(hidden_seq.mean(dim=1)).view(
            hidden_seq.shape[0], 1, hidden_seq.shape[-1]
        )
        att_scores = torch.bmm(query, hidden_seq.permute(0, 2, 1))
        predicted = torch.bmm(att_scores, hidden_seq).squeeze(1)
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

        for param in self.cls_model.parameters():
            param.requires_grad = False

        if reprog_front == "uni_noise":
            self.delta = torch.nn.Parameter(
                torch.Tensor(1, 1, 16000), requires_grad=True
            )
            torch.nn.init.xavier_uniform_(self.delta)
        elif reprog_front == "condi":
            n_channel = 50
            self.linear = nn.Linear(n_channel, 80)
            self.conv = nn.Sequential(
                nn.Conv1d(80, n_channel, 3, 1, 1),
                nn.ReLU(),
                # nn.BatchNorm1d(n_channel),
                # nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                # nn.ReLU(),
                # nn.BatchNorm1d(n_channel),
                # nn.Conv1d(n_channel, n_channel, 3, 1, 1),
                # nn.ReLU(),
                # nn.BatchNorm1d(n_channel),
                # nn.Conv1d(n_channel, n_channel, 3, 1, 1)
            )

        elif reprog_front == "mix":
            self.delta = torch.nn.Parameter(
                torch.Tensor(1, 1, 16000), requires_grad=True
            )
            self.linear = nn.Linear(80, 160)
            self.conv = nn.Sequential(
                nn.Conv1d(160, 160, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(160, 160, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(160, 80, 3, 1, 1),
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
        model.load_state_dict(ckpt["Model"])
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
                self.conv(
                    features.squeeze(1).permute(0, 2, 1)
                ).permute(0, 2, 1)
            ).unsqueeze(1)
            # features = self.linear_com(features)
        predicted = self.cls_model.transform(features)[
            :, : self.class_num * self.map_num
        ]

        predicted = predicted.view(
            -1,
            self.class_num,
            self.map_num
        ).sum(dim=-1)
        predicted = predicted.reshape(n_batch, -1, self.class_num).mean(1)
        return predicted
