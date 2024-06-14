import torch
import torch.nn as nn
import torchvision

from typing import *

from data import SimpleDataset

def get_model(model_str: str, pretrained=False):
    match (model_str.lower()):
        case "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            return efficientnet_b0() if not pretrained else efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
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

def get_distance_function(dist_func_str: str):
    match (dist_func_str.lower()):
        case "pairwise"|"l2"|"euclidean":
            return lambda x1,x2: nn.functional.pairwise_distance(x1,x2, p=2)
        case "cosine":
            return lambda x1,x2: -nn.functional.cosine_similarity(x1,x2,dim=1)
        case _:
            raise ValueError(f"Distance function `{dist_func_str}` is not supported.")

def generate_support_set(
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        n_way: int,
        k_shot: int,
        img_size: int,
        img_channels: int,
        device: Optional[str] = "cuda",
):
    data_tensor = torch.empty((n_way, k_shot, img_channels, img_size, img_size), device=device)
    for i in range(n_way):
        # Get all images of class `i`
        cls_idxs = torch.where(train_labels == i)
        class_imgs = train_images[cls_idxs]
        print(class_imgs.shape)

        # Select `k` random images of current class
        k_idxs = torch.randperm(class_imgs.size(0))[:k_shot]
        class_imgs = class_imgs[k_idxs]
        print(class_imgs.shape)
        class_imgs = class_imgs.view(k_shot,-1,img_size,img_size)
        data_tensor[i] = class_imgs
        print('---')

    return data_tensor
