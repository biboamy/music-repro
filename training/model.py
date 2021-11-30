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
from speechbrain.pretrained import EncoderClassifier


model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

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
                reprog=True,
                map_num=5,
                pad_num=500,
                dropout=0.4,
                ckpt='online',
                model_type='resnet18'):
        super(ImageModel, self).__init__()
        #if ckpt is not None:
        self.resnet = eval(model_type)(pretrained=True)
        #else:
        #    self.resnet = eval(model_type)(pretrained=False)
        self.reprog = reprog

        self.hcqt = HarmonicSTFT(sample_rate=sample_rate,
                                  n_fft=n_fft,
                                  n_harmonic=n_harmonic,
                                  semitone_scale=semitone_scale,
                                  learn_bw=learn_bw)

        self.preprocess = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(256),
            #torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.hcqt.parameters():
            param.requires_grad = False
        
        if reprog:
            self.delta = torch.nn.Parameter(torch.Tensor(1, 500), requires_grad=True)

            self.linear_emb = nn.Linear(152*3, pad_num*3)
            self.linear_noise = nn.Linear(500, pad_num*3*157)
            self.linear_com = nn.Sequential(
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(pad_num*3, pad_num*3, 3, 1, 1),
                nn.ReLU(),
                nn.Conv1d(pad_num*3, pad_num*3, 1),
            )

            self.leakyRely = nn.LeakyReLU()
            
        self.map_num = map_num
        self.class_num = n_class
        self.pad_num = pad_num

        torch.nn.init.xavier_uniform_(self.delta)


    def transform(self, features, isTrain=True, noise=None, **kwargs):
        x = self.hcqt(features) # min_max: (-100, 40)
        x = torch.clip((x + 100) / 121, 0, 1)

        if noise is not None:
            # 157 x 313
            n_batch, n_channel, n_freq, n_frame = x.shape
            x = self.linear_com((self.linear_emb(x.reshape(n_batch, -1, n_frame).permute(0, 2, 1)) +\
                                           self.leakyRely(self.linear_noise(self.delta)).reshape(n_frame, -1).unsqueeze(0).repeat(n_batch, 1, 1)).permute(0 ,2 ,1)).reshape(n_batch, 3, self.pad_num, n_frame)
            #x = F.pad(x, (0, 0, 0, self.pad_num-x.shape[-2]), "constant", 0)
            #x = (x) + (noise)
            #x = x + noise
        x = self.resnet(x)
        
        return x

    def forward(self, features, isTrain=True, **kwargs):
        noise = '1' #self.dropout(self.delta)

        predicted = self.transform(features, noise=noise)[:, :self.class_num * self.map_num] #self.classifier(self.transform(features, noise=noise))

        #predicted = self.transform(features, noise=noise)[:, :self.class_num * self.map_num]

        predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted


class VGGishModel(torch.nn.Module):
    def __init__(self, class_num=10, map_num=5, pad_num=500, dropout=0.4):
        super(VGGishModel, self).__init__()
        window_length_samples = int(round(16000 * 0.025))
        hop_length_samples = int(round(16000 * 0.01))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                n_fft=fft_length,
                hop_length=hop_length_samples,
                win_length=window_length_samples,
                n_mels=64,
                f_min=125,
                f_max=7500)
        self.vggish = VGGish(urls=model_urls, preprocess=False, postprocess=True, pretrained=True)
        for param in self.vggish.parameters():
            param.requires_grad = False

        self.class_num = class_num
        self.map_num = map_num

        self.classifier = torch.nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
        
        self.delta = torch.nn.Parameter(torch.Tensor(1, 1, 100, 70), requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.xavier_uniform_(self.delta)


    def transform(self, audio, isTrain=True, noise=None, **kwargs):
        if noise is not None:
            audio = F.pad(audio, (0, 70-audio.shape[-1], 0, 100-audio.shape[-2]), "constant", 0)
            audio = (audio) + (noise)

        #if isTrain:
        b_size, channel, time, freq = audio.shape
        audio = audio.reshape(-1, 1, 96, 64)
        audio = self.vggish(audio).reshape(b_size, channel, -1).mean(-2)

        predicted = self.classifier(audio)

        return predicted

    def forward(self, features, isTrain=True, **kwargs):
        noise = self.dropout(self.delta)

        predicted = self.transform(features, isTrain)#[:, :self.class_num * self.map_num] #self.classifier(self.transform(features, noise=noise))

        predicted = predicted#.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted


class SpeechModel(torch.nn.Module):
    def __init__(self, class_num=10, dropout=0.4, map_num=5):
        super(SpeechModel, self).__init__()
        self.class_num = class_num
        self.map_num = map_num
        run_opt_defaults = {
            "device": "cuda",
            "data_parallel_count": 1,
            "data_parallel_backend": True,
            "distributed_launch": False,
            "distributed_backend": "nccl",
            "jit_module_keys": None,
        }

        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts=run_opt_defaults, freeze_params=False).modules
        #for param in self.classifier.mean_var_norm.parameters():
        #    param.requires_grad = False
        #for param in self.classifier.embedding_model.parameters():
        #    param.requires_grad = False
        #for param in self.classifier.classifier.parameters():
        #    param.requires_grad = False
       

        self.delta = torch.nn.Parameter(torch.Tensor(1, 600, 80), requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)

        torch.nn.init.xavier_uniform_(self.delta)

    def transform(self, audio, isTrain, noise=None, **kwargs):

        feats = self.classifier.compute_features(audio)
        if noise is not None:
            feats = F.pad(feats, (0, 80-feats.shape[-1], 0, 600-feats.shape[-2]), "constant", 0)
            feats = feats + noise

        wav_lens = torch.ones(audio.shape[0], device=feats.device)
        feats = self.classifier.mean_var_norm(feats, wav_lens)
        embeddings = self.classifier.embedding_model(feats, wav_lens)
        predicted = self.classifier.classifier(embeddings).squeeze(1)
        return predicted

    def forward(self, features, isTrain=True, **kwargs):
        noise = self.dropout(self.delta)

        predicted = self.transform(features, isTrain, noise)[:, :self.class_num * self.map_num] #self.classifier(self.transform(features, noise=noise))

        predicted = predicted.view(-1, self.class_num, self.map_num).sum(dim=-1)
        return predicted