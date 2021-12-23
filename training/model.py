# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio

from modules import HarmonicSTFT
from torchvision.models import resnet18, resnet50, resnet101, resnet152, efficientnet_b7
import torchvision
from torchvggish.vggish import VGGish
from hubert import HubertForSequenceClassification
import hubert


model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

class Conv3_2d(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, kernal=3):
        super(Conv3_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernal, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out

class Conv3_2d_resmp(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2, kernal=3, padding=1):
        super(Conv3_2d_resmp, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, kernal, padding=padding)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, kernal, padding=padding)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        out = x + out
        out = self.mp(self.relu(out))
        return out

class CNNModel(torch.nn.Module):
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=1024,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=10,
                n_harmonic=3,
                semitone_scale=2,
                learn_bw='only_Q',
                model_type='CNN16k'):
        super(CNNModel, self).__init__()
        self.hcqt = HarmonicSTFT(sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_harmonic=n_harmonic,
                                  semitone_scale=semitone_scale,
                                  learn_bw=learn_bw)
        if model_type == 'CNN16k':
            conv_channels = 13
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, kernal=5),
                Conv3_2d(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels)
            )
            self.classifier = nn.Linear(1053, 10)

        if model_type == 'CNN235.5k':
            conv_channels = 58
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 2, kernal=6),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels)
            )
            self.classifier = nn.Linear(4698, 10)

        elif model_type == 'CNN14.1m':
            conv_channels = 126
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 1),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels*2, 2, kernal=5),
                Conv3_2d_resmp(conv_channels*2, conv_channels*2, (2, 1), kernal=5, padding=2),
                Conv3_2d_resmp(conv_channels*2, conv_channels*2, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(8064, 1016),
                nn.BatchNorm1d(1016),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1016, 10)
            )

        elif model_type == 'CNN14.4m':
            conv_channels = 128
            self.conv = nn.Sequential(
                Conv3_2d(3, conv_channels, 1),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d_resmp(conv_channels, conv_channels),
                Conv3_2d(conv_channels, conv_channels*2, 2, kernal=5),
                Conv3_2d_resmp(conv_channels*2, conv_channels*2, (2, 1), kernal=5, padding=2),
                Conv3_2d_resmp(conv_channels*2, conv_channels*2, 1)
            )
            self.classifier = nn.Sequential(
                nn.Linear(8192, 1006),
                nn.BatchNorm1d(1006),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1006, 10)
            )

    def forward(self, features, **kwargs):
        x = self.hcqt(features)
        x = self.conv(x).reshape(len(x), -1)
        #print(x.shape)
        x = self.classifier(x)

        return x


class ImageModel(torch.nn.Module):
    def __init__(self, 
                n_channels=128,
                sample_rate=16000,
                n_fft=1024,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=10,
                n_harmonic=3,
                semitone_scale=2,
                learn_bw='only_Q',
                map_num=5,
                pad_num=500,
                dropout=0.4,
                ckpt='online',
                model_type='resnet18', 
                reprog_front='None',
                fix_model=False):
        super(ImageModel, self).__init__()
        self.resnet = eval(model_type)(pretrained=True)
       
        self.hcqt = HarmonicSTFT(sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_harmonic=n_harmonic,
                                  semitone_scale=semitone_scale,
                                  learn_bw=learn_bw)


        if fix_model:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.hcqt.parameters():
                param.requires_grad = False
        
        if reprog_front == 'uni_noise':
            self.delta = torch.nn.Parameter(torch.Tensor(1, 3, pad_num, 157), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.delta)
        elif reprog_front == 'condi':
            self.linear_emb = nn.Linear(152*3, pad_num*3)
            self.linear_com = nn.Sequential(
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
            )
        elif reprog_front == 'mix':
            self.delta = torch.nn.Parameter(torch.Tensor(1, pad_num*3*157), requires_grad=True)
            self.linear_emb = nn.Linear(152*3, pad_num*3)
            self.linear_com = nn.Sequential(
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
            )
            torch.nn.init.xavier_uniform_(self.delta)
            
        self.map_num = map_num
        self.class_num = n_class
        self.pad_num = pad_num
        self.reprog_front = reprog_front


    def forward(self, features, **kwargs):
        x = self.hcqt(features) # min_max: (-100, 40)
        x = torch.clip((x + 100) / 121, 0, 1)

        # 157 x 313
        n_batch, n_channel, n_freq, n_frame = x.shape

        if self.reprog_front == 'uni_noise':
            # uni noise
            x = F.pad(x, (0, 0, 0, self.pad_num-x.shape[-2]), "constant", 0) + self.delta
        elif self.reprog_front == 'condi':
            # condi
            x = self.linear_com((self.linear_emb(x.reshape(n_batch, -1, n_frame).permute(0, 2, 1)) ).permute(0 ,2 ,1)).reshape(n_batch, 3, self.pad_num, n_frame)
        elif self.reprog_front == 'mix':
            # mix
            x = self.linear_com((self.linear_emb(x.reshape(n_batch, -1, n_frame).permute(0, 2, 1)) +self.delta.reshape(n_frame, -1).unsqueeze(0).repeat(n_batch, 1, 1)
                                           ).permute(0 ,2 ,1)).reshape(n_batch, 3, self.pad_num, n_frame)
        
        
        predicted = self.resnet(x)[:, :self.class_num * self.map_num]
        predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted


class VGGishModel(torch.nn.Module):
    def __init__(self, class_num=10, map_num=5, pad_num=500, dropout=0.4):
        super(VGGishModel, self).__init__()
        window_length_samples = int(round(16000 * 0.025))
        hop_length_samples = int(round(16000 * 0.01))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

        self.vggish = VGGish(urls=model_urls, preprocess=False, postprocess=True, pretrained=True)

        self.class_num = class_num
        self.map_num = map_num

        self.classifier = torch.nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
        self.delta = torch.nn.Parameter(torch.Tensor(1, 1, 96, 70), requires_grad=True)
        #self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.xavier_uniform_(self.delta)

        self.linear_emb = nn.Linear(64, 70)

        for param in self.vggish.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False


    def transform(self, audio, isTrain=True, noise=None, **kwargs):
            
        #if isTrain:
        b_size, channel, time, freq = audio.shape
        audio = audio.reshape(-1, 1, 96, 64)
        
        # uni noise
        #audio = F.pad(audio, (0, 70-audio.shape[-1]), "constant", 0) + self.delta
        
        # condi noise
        audio = F.pad(audio, (0, 70-audio.shape[-1]), "constant", 0) + self.linear_emb(audio)

        audio = self.vggish(audio).reshape(b_size, channel, -1).mean(-2)

        predicted = self.classifier(audio)

        return predicted

    def forward(self, features, isTrain=True, **kwargs):
        noise = '1' #(self.delta)

        predicted = self.transform(features, isTrain, noise)#[:, :self.class_num * self.map_num] #self.classifier(self.transform(features, noise=noise))

        predicted = predicted#.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted


class SpeechModel(torch.nn.Module):
    def __init__(self, class_num=10, dropout=0.4, map_num=5, fix_model=True, reprog_front='None'):
        super(SpeechModel, self).__init__()
        self.class_num = class_num
        self.map_num = map_num
        self.reprog_front = reprog_front
        
        self.model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")

        if fix_model:
            for name, param in self.model.named_parameters():
                if 'delta' not in name:
                    param.requires_grad = False
            
        self.delta = torch.nn.Parameter(torch.Tensor(1, 80000), requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.delta)

    def forward(self, input_values, attention_mask, **kwargs):

        if self.reprog_front == 'uni_noise':
            #feats = F.pad(feats, (0, 80-feats.shape[-1], 0, 600-feats.shape[-2]), "constant", 0)
            input_values = input_values #+ torch.nn.Identity()(self.delta)

        predicted = self.model(input_values=input_values).logits
        predicted = predicted[:, :self.class_num*self.map_num]
        predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted
