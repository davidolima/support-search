"""
David Lima, 2024
"""

import random
from typing import *
import warnings

import torch
from torch.utils.data import DataLoader
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
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            warnings.warn(f"Device not specified for PrototypicalNetwork. Using {device}.")

        self.backbone = backbone
        self.distance_function = distance_function
        self.use_softmax = use_softmax
        self.n_classes = n_classes

        self.backbone.to(device)
        self.prototypes = self._compute_prototypes(support_set, device=device)

    def _compute_prototypes(self, support_set: SupportSet, device=None) -> torch.Tensor:
        """
        Params:
          support_set: SupportSet
          device: Optional[str] = None - Device to be used.
        Returns:
          A tensor of shape (N_WAY, N_FEATURES) containing the mean of the features of each class.
        """
        self.backbone.eval()
        x = self.backbone(support_set[0][0].unsqueeze_(0).to(device))
        assert len(x.shape) > 1, f"Expected backbone to return a 1D tensor of features, got tensor of shape {x.shape} instead."

        support_set_features = torch.empty((support_set.n_way*support_set.k_shot, x.size(1)), device=device)
        for i, (x, _) in enumerate(support_set):
            x = x.to(device).unsqueeze_(0)
            support_set_features[i] = self.backbone(x)

        self.backbone.train()
        return torch.cat([
            support_set_features[torch.nonzero(support_set.targets == y)].mean(0) for y in range(support_set.n_way)
        ])

    def _compute_distance_to_prototypes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
          x: torch.Tensor - Vector of shape (BATCH_SIZE, N_FEATURES)
        Returns:
          A tensor of shape (BATCH_SIZE, N_WAY) containing the distance of `x` between each prototype.
        """
        # TODO
        raise RuntimeError("Not implemented yet.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self._compute_distance_to_prototypes(x)
        if self.use_softmax:
            x = x.softmax(dim=0)
        return x

    def predict(self, dataloader: DataLoader) -> Tuple[float]:
        # TODO
        self.backbone.eval()
        for (x,y) in dataloader:
            preds = self.forward(x)
            print(preds, y)
            break
        self.backbone.train()
        raise RuntimeError("Not implemented yet.")
