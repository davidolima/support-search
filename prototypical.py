import math
import random
import warnings
from enum import Enum
from typing import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from tqdm import tqdm

from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score

from support_set import SupportSet
from utils import *

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
            device: Optional[Union[str, torch.device]] = None,
            class_wise: Optional[bool] = False,
    ) -> Tuple:
        """
        Params:
          dataloader: DataLoader object for desired dataset.
          device: Device on which the calculations will be performed.
        Returns:
          A tuple of floats, containing the model's accuracy, f1_score, precision
          and recall calculated during the evaluation.
        """
        assert self.n_way, "Unreachable: PrototypicalNetwork.n_way should be already be defined by now."

        device = get_device() if device is None else device

        performance_metrics = torch.zeros((4), device=device) # tp, fp, fn, tn
        class_wise_metrics = torch.zeros((self.n_way, 4), device=device) # tp, fp, fn, tn

        self.backbone.to(device).eval()
        for (x,y) in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                distances = self.forward(x)
            preds = distances.argmin(1)

            performance_metrics += multiclass_performance_metrics(
                y_pred=preds,
                y_true=y,
                classes=self.n_way,
                device=device
            )

            class_wise_metrics += class_wise_performance_metrics(
                y_pred=preds,
                y_true=y,
                classes=self.n_way,
                device=device
            )

        self.backbone.train()

        return class_wise_metrics if class_wise else (
            calculate_accuracy_score(performance_metrics),
            calculate_f1_score(performance_metrics),
            calculate_precision(performance_metrics),
            calculate_recall(performance_metrics),
        )
