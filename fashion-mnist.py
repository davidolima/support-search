#!/usr/bin/env python3

import argparse
from typing import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T

from prototypical import PrototypicalNetwork
from support_set import SupportSet
from utils import *

# TODO: Get rid of this.
NUMBER_OF_IMAGE_PERMUTATIONS = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script for search the k most representative images for each class of a given dataset set.",
    )

    parser.add_argument("--device", type=str)
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--dist-function", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epochs-per-search", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()
    if not args.device: args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[!] Current configuration:")
    [print(f"      {key}: {value}") for key,value in args.__dict__.items()]

    backbone, preprocessing = get_model(args.backbone)

    optimizer = get_optimizer(args.optimizer)(backbone.parameters(), lr=args.lr)
    dataset = FashionMNIST(
        root='/datasets/fashion-mnist/',
        train=True,
        download=True,
        transform=preprocessing,
        # transform=T.Compose([
        #     T.Grayscale(num_output_channels=args.input_channels),
        #     T.ToTensor(),
        #     T.Normalize(mean=[.5]*args.input_channels, std=[.5]*args.input_channels),
        # ]),
    )

    for _ in range(NUMBER_OF_IMAGE_PERMUTATIONS):
        support_set = SupportSet.random_from_tensor(
            train_images=dataset.data,
            train_labels=dataset.targets,
            img_size=args.input_size,
            img_channels=args.input_channels,
            n_way=5,
            k_shot=5,
            device=args.device,
        )

        model = PrototypicalNetwork(
            backbone=backbone,
            support_set=support_set,
            distance_function=args.dist_function,
            use_softmax=False,
            device=args.device,
        )

        model.predict(
            dataloader=DataLoader(
                dataset=support_set,
                batch_size=args.batch_size,
                shuffle=True,
            )
        )
