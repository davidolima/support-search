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

from support_set import SupportSet
import math

class PrototypicalNetwork(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            distance_function: Callable,
            support_set: SupportSet,
            use_softmax: Optional[bool] = False,
            device: Optional[str] = None,
            n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.distance_function = distance_function
        self.use_softmax = use_softmax
        self.n_classes = n_classes

        self.prototypes = self._compute_prototypes(support_set, device=device)

    def extract_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.backbone(input)

    def apply_distance_function(self, x) -> torch.Tensor:
        return self.distance_function(x)

    def _compute_prototypes(self, support_set: SupportSet, device=None) -> torch.Tensor:
        #TODO: Fix this crap
        sample = self.backbone(support_set[0][0])
        #assert len(sample.shape) > 1, f"Expected backbone to return a 1D tensor of features, got tensor of shape {x.shape} instead."

        support_set_features = torch.empty((support_set.n_way, support_set.k_shot, sample.size(1)), device=device)
        for i, (x, y) in enumerate(support_set):
            support_set_features[math.floor(i/support_set.n_way)][i%support_set.k_shot] = self.extract_features(x)

        return support_set_features

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
