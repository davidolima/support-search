import torch
import torch.nn as nn
import torchvision

from typing import *

from data import SimpleDataset

def get_model(model_str: str, pretrained=False) -> Tuple[nn.Module, torchvision.transforms.Compose]:
    match (model_str.lower()):
        case "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            return (efficientnet_b0() if not pretrained else efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)), EfficientNet_B0_Weights.DEFAULT.transforms()
        case _:
            raise ValueError(f"Model `{model_str}` is not supported.")

def get_optimizer(optim_str: str):
    match (optim_str.lower()):
        case "adam":
            from torch.optim import Adam
            return Adam
        case "adamw":
            from torch.optim import AdamW
            return AdamW
        case _:
            raise ValueError(f"Optimizer `{optim_str}` is not supported.")
