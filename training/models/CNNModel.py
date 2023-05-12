# coding: utf-8
import torch
import torch.nn as nn
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
        if model_type == "CNN16k":
            conv_channels = 10  # 6 #10
            linear_channels = 810  # 486 #810
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, kernal=5),
                Conv3_2d(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels),
            )
            self.classifier = nn.Linear(linear_channels, n_class)
            # 1053, 13
            # 810, 10

        if model_type == "CNN235.5k":

            conv_channels = 54
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 2, kernal=3),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
            )
            self.classifier = nn.Linear(9234, n_class)
            # 4698 58
            # 4374 54

        elif model_type == "CNN14.1m":
            conv_channels = 126
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 1),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels * 2, 2, kernal=5),
                Conv3_2d_resmp(
                    conv_channels * 2, conv_channels * 2, (2, 1), kernal=5, padding=2
                ),
                Conv3_2d_resmp(conv_channels * 2, conv_channels * 2, 1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(8064, 1007),
                nn.BatchNorm1d(1007),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1007, n_class),
            )
            # 1016
            # 1007

        elif model_type == "CNN14.4m":
            conv_channels = 128
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 1),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels * 2, 2, kernal=5),
                Conv3_2d_resmp(
                    conv_channels * 2, conv_channels * 2, (2, 1), kernal=5, padding=2
                ),
                Conv3_2d_resmp(conv_channels * 2, conv_channels * 2, 1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(8192, 1004),
                nn.BatchNorm1d(1004),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1004, n_class),
            )
            # 1006
            # 1004

    def forward(self, features, **kwargs):
        x = self.hcqt(features)
        x = self.conv(x).reshape(len(x), -1)
        # print(x.shape)
        x = self.classifier(x)

        return x
