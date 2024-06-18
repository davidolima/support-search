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
