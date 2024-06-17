"""
David Lima, 2024
"""

import random
from typing import *
import warnings

import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision.datasets import VisionDataset

class PrototypicalNetwork(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            distance_function: Callable,
            use_softmax: Optional[bool] = False,
            device: Optional[str] = None,
            n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.distance_function = distance_function
        self.use_softmax = use_softmax
        self.n_classes = n_classes

    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.backbone(input)

    def apply_distance_function(self, x) -> torch.Tensor:
        return self.distance_function(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.apply_distance_function(x)
        if self.use_softmax:
            x = x.softmax(dim=0)
        return x

    @staticmethod
    def get_random_images_from_class(class_idxs: torch.Tensor, dataset, in_channels, in_size):
        idxs = torch.zeros((class_idxs.size(0), in_channels, in_size, in_size))
        for i in range(class_idxs.size(0)):
            possible_idxs = [j for j in range(len(dataset.targets)) if dataset.targets[j] == class_idxs[i]]
            idxs[i] = dataset[random.choice(possible_idxs)][0]
        return idxs

    def train_step(
            self,
            criterion,
            optimizer,
            dataset,
            device,
        ) -> float:
        assert self.n_classes is not None, "Something went wrong; Number of classes is unknown."

        running_loss = 0
        total = 0

        for i, (x, y) in enumerate(dataset):
            x, y = x.to(device), y.to(device)

            negative_idx = torch.empty((x.size(0)), device=device)
            for i in range(y.size(0)):
                possible_idxs = list(range(self.n_classes))
                possible_idxs.remove(y[i])
                negative_idx[i] = random.choice(possible_idxs)
                del possible_idxs

            positives = self.get_random_images_from_class(y, dataset, x.shape[1], x.shape[2]).to(device)
            negatives = self.get_random_images_from_class(negative_idx, dataset, x.shape[1], x.shape[2]).to(device)

            anchors = self.extract_features(x)
            positives = self.extract_features(positives)
            negatives = self.extract_features(negatives)

            loss = criterion(anchors, positives, negatives)

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return running_loss/total

    def fit(
        self,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        dataset: VisionDataset,
        device: Optional[str] = None,
    ):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else "cpu"
            warnings.warn(f"[!] Device not specified. Will proceed on detected device: {device}")
        if not self.n_classes:
            self.n_classes = len(torch.unique(dataset.targets))
            warnings.warn(f"[!] Detected {self.n_classes} classes.")

        self.backbone.train().to(device)
        running_loss = 0
        for epoch in tqdm(range(epochs)):
            running_loss += self.train_step(
                dataset=dataset,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            print(f"[{epoch}/{epochs}] Loss: {running_loss/((epoch+1)*len(dataset)):.2f}")
