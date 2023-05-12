# coding: utf-8
import numpy as np
import tqdm
import argparse
from sklearn import metrics

import torch
from torch.autograd import Variable


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
        if self.dataset == "gtzan":
            from data_loader.gtzan_loader import get_audio_loader

            self.n_classes = 10
        elif self.dataset == "FMA":
            from data_loader.FMA_loader import get_audio_loader

            self.n_classes = 16
        elif self.dataset == "singer32":
            from data_loader.singer32_loader import get_audio_loader

            self.n_classes = 32
        self.loader = get_audio_loader(
            config.data_path,
            1,
            "test",
            config.num_workers,
            input_length=config.input_length,
        )
        self.build_model()

    def get_model(self):
        if self.model_type in ["CNN235.5k"]:
            from models.CNNModel import CNNModel

            return CNNModel(model_type=self.model_type, n_class=self.n_classes)
        elif self.model_type == "speechatt":
            from models.SpeechModel import V2SReprogModel

            return V2SReprogModel(
                map_num=self.map_num,
                n_class=self.n_classes,
                reprog_front=self.reprog_front,
            )
        elif self.model_type == "ast":
            from models.ASTModel import AST

            return AST(n_class=self.n_classes, reprog_front=self.reprog_front)
        else:
            print(
                "model_type has to be one of [CNN235.5k, speechatt, ast]"
            )

    def build_model(self):
        self.model = self.get_model()

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def load(self, filename):
        S = torch.load(filename)
        if "spec.mel_scale.fb" in S.keys():
            self.model.spec.mel_scale.fb = S["spec.mel_scale.fb"]
        self.model.load_state_dict(S)

    def to_var(self, x):
        if isinstance(x, dict):
            for d in x.keys():
                if self.is_cuda:
                    x[d] = Variable(x[d]).cuda().squeeze()
            return x
        else:
            if self.is_cuda:
                x = x.cuda()
            return Variable(x).squeeze()

    def get_auc(self, est_array, gt_array):
        from scipy.special import softmax

        est_array = softmax(est_array, 1)
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array,
                                                  est_array, average="macro")
        return roc_aucs, pr_aucs

    def get_score(self, est_array, gt_array):
        acc = metrics.accuracy_score(
            np.argmax(gt_array, axis=1), np.argmax(est_array, axis=1)
        )

        return acc

    def test(self):
        roc_auc, pr_auc, acc, loss = self.get_test_score()
        print("loss: %.4f" % loss)
        print("roc_auc: %.4f" % roc_auc)
        print("pr_auc: %.4f" % pr_auc)
        print("acc: %.4f" % acc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        for x, y in tqdm.tqdm(self.loader):
            # forward
            x = self.to_var(x)
            if "resnet" in self.model_type:
                y = self.to_var(y).repeat(len(x), 1)
            elif "CNN" in self.model_type:
                y = self.to_var(y).repeat(len(x), 1)
            elif self.model_type == "speechatt":
                y = self.to_var(y).repeat(len(x), 1)
            elif self.model_type == "ast":
                y = self.to_var(y).repeat(len(x), 1)
            elif self.model_type == "hubert_ks":
                y = self.to_var(y).repeat(len(x["input_values"]), 1)

            if x.shape[0] > self.batch_size:
                out = np.zeros((x.shape[0], self.n_classes))
                mini_batch = int(np.ceil(x.shape[0] / self.batch_size))
                for i in range(mini_batch):
                    out[
                        i * self.batch_size: i * self.batch_size
                        + self.batch_size
                        ] = (
                        self.model(
                            x[
                                i * self.batch_size: i * self.batch_size
                                + self.batch_size
                            ]
                        )
                        .detach()
                        .cpu()
                    )
            else:
                out = self.model(x).detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(y.detach().cpu().numpy()[0])

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = 0  # np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        acc = self.get_score(est_array, gt_array)
        return roc_auc, pr_auc, acc, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gtzan",
        choices=["gtzan", "FMA", "singer32"]
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet101",
        choices=[
            "resnet18",
            "resnet50",
            "resnet101",
            "resnet152",
            "efficientnet_b7",
            "CNN16k",
            "CNN235.5k",
            "CNN14.1m",
            "CNN14.4m",
            "vggish",
            "hubert_ks",
            "speechatt",
            "ast",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_load_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--map_num", type=int, default=5)
    parser.add_argument("--pad_num", type=int, default=500)
    parser.add_argument(
        "--reprog_front",
        type=str,
        default=["None"],
        choices=["None", "uni_noise", "condi", "mix"],
    )
    parser.add_argument("--input_length", type=int, default=80000)

    config = parser.parse_args()

    p = Predict(config)
    p.test()
