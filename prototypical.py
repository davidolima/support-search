from enum import Enum
import random
from typing import *
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torchvision.datasets import VisionDataset

from support_set import SupportSet
from utils import get_device
import math

_supported_distance_functions = Enum("_supported_distance_functions", [
    "cosine","cos",
    "l2","euclidean","pairwise"
])

class PrototypicalNetwork(nn.Module):
    def __init__(
            self,
            backbone: nn.Module,
            distance_function: _supported_distance_functions,
            support_set: SupportSet,
            use_softmax: Optional[bool] = False,
            device: Optional[Union[str, torch.device]] = None,
            n_way: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.backbone.to(get_device() if device is None else device)

        self.distance_function = PrototypicalNetwork.get_distance_function(distance_function)
        self.use_softmax = use_softmax
        self.n_way = n_way

        self.prototypes = self._compute_prototypes(support_set, device=device)

    def _compute_prototypes(self, support_set: SupportSet, device: Optional[Union[str, torch.device]] =None) -> torch.Tensor:
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

        self.n_way = support_set.n_way #TODO: Refactor this

        support_set_features = torch.empty((support_set.n_way*support_set.k_shot, x.size(1)), device=device)
        for i, (x, _) in enumerate(support_set):
            x = x.to(device).unsqueeze_(0)
            support_set_features[i] = self.backbone(x)

        self.backbone.train()
        return torch.cat([
            support_set_features[torch.nonzero(support_set.targets == y)].mean(0) for y in range(support_set.n_way)
        ])

    @staticmethod
    def get_distance_function(fn: Union[Callable, _supported_distance_functions]):
        if callable(fn):
            return fn
        elif fn in ('cos', 'cosine'):
            return lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=0)
        elif fn in ('l2', 'euclidean', 'pairwise'):
            return nn.functional.pairwise_distance
        else:
            raise ValueError(f"Unexpected value for `distance_function`: {fn}. Expected: {_supported_distance_functions._member_names_}")

    def _compute_distance_to_prototypes(self, input: torch.Tensor) -> torch.Tensor:
        """
        Params:
          input: torch.Tensor - Vector of features of a single image.
        Returns:
          A tensor of shape (N_WAY) containing the distance between `input` and each class prototype.
        """
        return torch.cat([
            self.distance_function(input, prototype).unsqueeze_(0) for prototype in self.prototypes
        ])

    def _batch_compute_distance_to_prototypes(self, input: torch.Tensor, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """
        Params:
          input: torch.Tensor - Vector of shape (BATCH_SIZE, N_FEATURES).
        Returns:
          A tensor of shape (BATCH_SIZE, N_WAY), where each x_{ij} is the distance between the i-th batch image to the j-th class prototype.
        """
        assert self.n_way, "Unreachable: `n_way` should already be defined here."
        assert len(input.shape) == 2, f"Expected input to vector to be 2-dimensional, got {input.shape}."

        device = get_device() if device is None else device
        distances = torch.zeros((input.size(0), self.n_way), device=device)

        for i in range(input.size(0)):
            distances[i] = self._compute_distance_to_prototypes(input[i])

        return distances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self._batch_compute_distance_to_prototypes(x, device=x.device)
        return x.softmax(dim=0) if self.use_softmax else x

    def evaluate(
            self,
            dataloader: DataLoader,
            device: Optional[Union[str, torch.device]] = None
    ) -> Tuple:
        """
        Params:
          dataloader: DataLoader object for desired dataset.
          device: Device on which the calculations will be performed.
        Returns:
          A tuple of floats, containing the model's accuracy, f1_score, precision
          and recall calculated during the evaluation.
        """
        device = get_device() if device is None else device

        metrics = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

        self.backbone.to(device).eval()
        for (x,y) in dataloader:
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                distances = self.forward(x)
            preds = distances.argmin(1)

            for cls in range(self.n_way):
                binary_preds = (preds == cls)*cls
                binary_true = (y == cls)*cls
                metrics["tp"] += ((binary_preds == 1) & (binary_true == 1)).sum().item()
                metrics["fp"] += ((binary_preds == 1) & (binary_true == 0)).sum().item()
                metrics["fn"] += ((binary_preds == 0) & (binary_true == 1)).sum().item()
                metrics["tn"] += ((binary_preds == 0) & (binary_true == 0)).sum().item()

        acc       = (metrics['tp']+metrics['tn'])/(metrics['tp']+metrics['fp']+metrics['fn']+metrics['tn']) if (metrics['tp']+metrics['fp']+metrics['fn']+metrics['tn']) > 0 else .0
        precision = metrics['tp']/(metrics['tp']+metrics['fp']) if (metrics['tp']+metrics['fp']) > 0 else .0
        recall    = metrics['tp']/(metrics['tp']+metrics['fn']) if (metrics['tp']+metrics['fn']) > 0 else .0
        f1_score  = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else .0

        self.backbone.train()
        return acc, f1_score, precision, recall
