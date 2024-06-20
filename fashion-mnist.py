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
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--support-set-permutations", type=int, default=5)

    args = parser.parse_args()
    if not args.device: args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[!] Current configuration:")
    [print(f"      {key}: {value}") for key,value in args.__dict__.items()]

    backbone, preprocessing = get_model(args.backbone)

    checkpoint = torch.load("./best_f1")
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['state_dict'], strict=False)
    print("[!] Checkpoint Loaded Successfully.")

    optimizer = get_optimizer(args.optimizer)(backbone.parameters(), lr=args.lr)
    dataset = FashionMNIST(
        root='/datasets/fashion-mnist/',
        train=True,
        download=True,
        transform=preprocessing,
    )

    for i in range(args.support_set_permutations):
        # Create support set from random image of dataset
        support_set = SupportSet.random_from_tensor(
            train_images=dataset.data,
            train_labels=dataset.targets,
            img_size=args.input_size,
            img_channels=args.input_channels,
            n_way=args.n_classes,
            k_shot=5,
            device=args.device,
        )

        # Initialize model with current support set
        model = PrototypicalNetwork(
            backbone=backbone,
            support_set=support_set,
            distance_function=args.dist_function,
            use_softmax=False,
            device=args.device,
        )

        # Evaluate the model's classification performance on test set
        class_wise_metrics = model.evaluate(
            dataloader=DataLoader(
                dataset=support_set, # TODO: CHANGE TO TEST SET!
                batch_size=args.batch_size,
                shuffle=True,
            ),
            class_wise = True,
        )

        print("Class-wise metrics:")
        for cls in range(args.n_classes):
            cls_f1        = calculate_f1_score(class_wise_metrics[cls])
            cls_acc       = calculate_accuracy_score(class_wise_metrics[cls])
            cls_precision = calculate_precision(class_wise_metrics[cls])
            cls_recall    = calculate_recall(class_wise_metrics[cls])
            print(f"  {cls}: acc {cls_acc:.2f}% f1-score {cls_f1:.2f}% precision {cls_precision:.2f}% recall {cls_recall:.2f}%")

        #print(f"[SupportSet {i}/{args.support_set_permutations}] Accuracy: {accuracy:.2f}% F1-Score: {f1_score:.2f}% Precision: {precision:.2f}% Recall: {recall:.2f}%")
