# coding: utf-8
import os
import time
import numpy as np
from sklearn import metrics
import datetime
import sys
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import soundfile as sf

import model as Model


class Solver(object):
    def __init__(self, data_loader, valid_loader, config):
        # data loader
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard
        self.map_num = config.map_num
        self.pad_num = config.pad_num
        self.reprog_front = config.reprog_front
        self.fix_model = config.fix_model

        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.get_dataset()
        self.build_model()

        #self.load('../models/vggFix/best_model.pth')

        # Tensorboard
        self.writer = SummaryWriter()

    def get_dataset(self):
        if self.dataset == 'gtzan':
            self.valid_list = open(f"{self.data_path}/valid_filtered.txt", 'r').readlines()
            self.mappeing = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}


    def get_model(self):
        if self.model_type in ['resnet18', 'resnet50', 'resnet101', 'efficientnet_b7', 'resnet152']:
            return Model.ImageModel(model_type=self.model_type, map_num=self.map_num, pad_num=self.pad_num, reprog_front=self.reprog_front, fix_model=self.fix_model)
        elif self.model_type == 'vggish':
            return Model.VGGishModel(map_num=self.map_num)
        elif self.model_type in ['hubert_ks']:
            return Model.SpeechModel(map_num=self.map_num, fix_model=self.fix_model, reprog_front=self.reprog_front)

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def to_var(self, x):
        if isinstance(x, dict):
            for d in x.keys():
                if torch.cuda.is_available():
                    x[d] = Variable(x[d]).cuda().squeeze()
            return x
        else:
            if torch.cuda.is_available():
                x = x.cuda()
            return Variable(x)

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
           
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                b_size = len(x)
                x = self.to_var(x)
                y = self.to_var(y)

                out = self.model(x['input_values'], x['attention_mask'])
                # Backward
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def load(self, filename):
        S = torch.load(filename)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in S.items() if k in model_dict and 'delta' not in k}
        model_dict.update(pretrained_dict) 
        self.model.load_state_dict(model_dict)


    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001,
                                            momentum=0.9, weight_decay=0.0001,
                                            nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 50:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0005
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 5e-4')
        '''
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        '''
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)


    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        print('roc_auc: %.4f pr_auc: %.4f' % (roc_aucs, pr_aucs))
        return roc_aucs, pr_aucs

    def get_acc(self, est_array, gt_array):
        #est_array[est_array>0.5] = 1
        #est_array[est_array<=0.5] = 0
        acc = metrics.accuracy_score(gt_array, est_array)
        print('Acc: %.4f' % acc)
        return acc

    def print_log(self, epoch, ctr, loss_bce, start_t):
        if (ctr) % self.log_step == 0:
            sys.stdout.write('\r')
            sys.stdout.write("[%s] Epoch [%d/%d] Iter [%d/%d] train bce loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.data_loader), loss_bce.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))
            sys.stdout.flush()


    def validation(self, best_metric, epoch):
        acc, loss = self.get_validation_score(epoch)
        score = acc
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric


    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        with torch.no_grad():
            est_array = []
            gt_array = []
            losses = []
            reconst_loss = self.get_loss_function()
            index = 0
            for x, y in self.valid_loader:
                # forward
                x = self.to_var(x)

                if 'resnet' in self.model_type:
                    y = self.to_var(y).repeat(len(x), 1)
                    out = self.model(x)
                elif self.model_type == 'hubert_ks':
                    y = self.to_var(y).repeat(len(x['input_values']), 1)
                    chunk_size = int(np.ceil(len(y) / self.batch_size))
                    out = torch.empty(y.shape, device = 'cuda')
                    for c in range(chunk_size):
                        inp = x['input_values'][c*self.batch_size:(c+1)*self.batch_size]
                        att = x['attention_mask'][c*self.batch_size:(c+1)*self.batch_size]
                        out[c*self.batch_size:(c+1)*self.batch_size] = self.model(input_values=inp, attention_mask=att)
                
                
                loss = reconst_loss(out, y)
                losses.append(float(loss.data))
                out = out.detach().cpu()

                # estimate
                estimated = np.array(out).mean(axis=0)
                est_array.append(estimated)

                gt_array.append(y.detach().cpu().numpy()[0])
                index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        acc = self.get_acc(np.argmax(est_array, axis=1), np.argmax(gt_array, axis=1))
        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        self.writer.add_scalar('Loss/valid', loss, epoch)
        #self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        #self.writer.add_scalar('AUC/PR', pr_auc, epoch)
        self.writer.add_scalar('ACC', acc, epoch)
        return acc, loss

