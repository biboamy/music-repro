import os
import argparse
from solver import Solver
import torch
import numpy as np
import random


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    fix_seed(1234)

    # import data loader
    if config.dataset == "gtzan":
        from data_loader.gtzan_loader import get_audio_loader

    # get data loder
    train_loader = get_audio_loader(
        config.data_path,
        config.batch_size,
        split="TRAIN",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    valid_loader = get_audio_loader(
        config.data_path,
        1,
        split="valid",
        input_length=config.input_length,
        num_workers=config.num_workers,
    )
    solver = Solver(train_loader, valid_loader, config)
    solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument(
        "--dataset",
        type=str,
        default="gtzan",
        choices=["gtzan"]
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ast",
        choices=["CNN235.5k", "speechatt", "ast"],
    )
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use_tensorboard", type=int, default=1)
    parser.add_argument("--model_save_path", type=str, default="../models")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--log_step", type=int, default=20)
    parser.add_argument("--input_length", type=int, default=160000)
    parser.add_argument("--map_num", type=int, default=5)
    parser.add_argument("--pad_num", type=int, default=500)
    parser.add_argument(
        "--reprog_front",
        type=str,
        default=["None"],
        choices=["None", "uni_noise", "condi", "skip"],
    )

    config = parser.parse_args()

    print(config)
    main(config)
