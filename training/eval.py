# coding: utf-8
import os
import time
import numpy as np
import datetime
import tqdm
import argparse
import pickle
from sklearn import metrics

import torch
import torch.nn as nn
from torch.autograd import Variable
import soundfile as sf
from data_loader.gtzan_loader import get_audio_loader

import model as Model


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.map_num = config.map_num
        self.pad_num = config.pad_num
        self.reprog_front = config.reprog_front
        self.loader = get_audio_loader(config.data_path, 1, 'test', config.num_workers, input_length=80000)
        self.build_model()
        self.get_dataset()

    def get_model(self):
        if self.model_type in ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b7', 'resnet152']:
            self.input_length = 80000
            return Model.ImageModel(model_type=self.model_type, map_num=self.map_num, pad_num=self.pad_num, reprog_front=self.reprog_front)
        elif self.model_type == 'vggish':
            return Model.VGGishModel(map_num=self.map_num)
        elif self.model_type in ['lang_ecapa']:
            return Model.SpeechModel(map_num=self.map_num)
        else:
            print('model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]')

    def build_model(self):
        self.model = self.get_model()

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()
            
    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def get_dataset(self):
        if self.dataset == 'gtzan':
            self.test_list = open(f"../../Voice2Series-Reprogramming-Pytorch-main/Datasets/gtzan/test_filtered.txt", 'r').readlines()
            self.mappeing = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

    def to_var(self, x):
        if isinstance(x, dict):
            for d in x.keys():
                if torch.cuda.is_available():
                    x[d] = Variable(x[d]).cuda().squeeze()
            return x
        else:
            if torch.cuda.is_available():
                x = x.cuda()
            return Variable(x).squeeze()

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'gtzan':
            npy_path = os.path.join(self.data_path, 'GTZAN', 'genres', fn)

        try:
            raw = np.load(npy_path, mmap_mode='r')
        except:
            raw = sf.read(npy_path)[0]


        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def get_score(self, est_array, gt_array):
        acc = metrics.accuracy_score(np.argmax(gt_array, axis=1), np.argmax(est_array, axis=1))

        return acc

    def test(self):
        roc_auc, pr_auc, acc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)
        print('acc: %.4f' % acc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCEWithLogitsLoss()
        for x, y in self.loader:

            # forward
            x = self.to_var(x)
            if 'resnet' in self.model_type:
                y = self.to_var(y).repeat(len(x), 1)
            elif 'CNN' in self.model_type:
                y = self.to_var(y).repeat(len(x), 1)
            elif self.model_type == 'hubert_ks':
                y = self.to_var(y).repeat(len(x['input_values']), 1)

            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(y.detach().cpu().numpy()[0])

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        acc = self.get_score(est_array, gt_array)
        return roc_auc, pr_auc, acc, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='gtzan', choices=['gtzan'])
    parser.add_argument('--model_type', type=str, default='resnet101',
                        choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_b7', 'vggish', 'hubert_ks', 'resnet152'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--map_num', type=int, default=5)
    parser.add_argument('--pad_num', type=int, default=500)
    parser.add_argument('--reprog_front', type=str, default=['None'], choices=['None', 'uni_noise', 'condi', 'mix'])

    config = parser.parse_args()

    p = Predict(config)
    p.test()






