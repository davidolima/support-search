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

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def single_class_performance_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        cls: int,
        device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    y_pred, y_true = y_pred.to(device), y_true.to(device)

    binary_preds = (y_pred == cls)
    binary_true = (y_true == cls)

    tp = ((binary_preds == 1) & (binary_true == 1)).sum().unsqueeze_(0)
    fp = ((binary_preds == 1) & (binary_true == 0)).sum().unsqueeze_(0)
    fn = ((binary_preds == 0) & (binary_true == 1)).sum().unsqueeze_(0)
    tn = ((binary_preds == 0) & (binary_true == 0)).sum().unsqueeze_(0)

    return torch.cat([tp,fp,fn,tn])

def multiclass_performance_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        classes: int,
        device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    metrics = torch.zeros((4), device=device) # tp, fp, fn, tn
    for cls in range(classes):
        metrics.add_(single_class_performance_metrics(y_pred, y_true, cls, device=device))
    return metrics

def class_wise_performance_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        classes: int,
        device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    metrics = torch.zeros((classes, 4), device=device)  # tp, fp, fn, tn
    for cls in range(classes):
        metrics[cls] = single_class_performance_metrics(y_pred, y_true, cls, device=device)
    return metrics

def calculate_accuracy_score(
        performance_metrics: torch.Tensor,
) -> torch.Tensor:
    """
    Params:
      performance_metrics: torch.Tensor of shape (4) containing the True Positives (tp),
                           False Positives (fp), False Negatives (fn) and True Negatives (tn)
                           count.
    Returns:
      Result of the equation (tp+tn) / (tp+tn+fp+fn)
    """
    tp, _, _, tn = performance_metrics
    return (tp+tn)/(performance_metrics.sum())

def calculate_precision(
        performance_metrics: torch.Tensor,
) -> torch.Tensor:
    """
    Params:
      performance_metrics: torch.Tensor of shape (4) containing the True Positives (tp),
                           False Positives (fp), False Negatives (fn) and True Negatives (tn)
                           count.
    Returns:
      Result of the equation (tp) / (tp+fp)
    """
    tp, fp, _, _ = performance_metrics
    return  tp / (tp + fp)

def calculate_recall(
        performance_metrics: torch.Tensor,
) -> torch.Tensor:
    """
    Params:
      performance_metrics: torch.Tensor of shape (4) containing the True Positives (tp),
                           False Positives (fp), False Negatives (fn) and True Negatives (tn)
                           count.
    Returns:
      Result of the equation (tp) / (tp+fn)
    """
    tp, _, fn, _ = performance_metrics
    return  tp / (tp + fn)

def calculate_f1_score(
        performance_metrics: torch.Tensor,
) -> torch.Tensor:
    """
    Params:
      performance_metrics: torch.Tensor of shape (4) containing the True Positives (tp),
                           False Positives (fp), False Negatives (fn) and True Negatives (tn)
                           count.
    Returns:
      Result of the equation 2*precision*recall/(precision+recall)
    """
    precision = calculate_precision(performance_metrics)
    recall = calculate_recall(performance_metrics)
    return 2*precision*recall/(precision+recall)
