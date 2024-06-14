import torch
from torchvision.datasets import DatasetFolder
from typing import *

class SimpleDataset:
    def __init__(
            self,
            data: str,
            transform: Optional[Callable] = None,
    ) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, index: int) -> Any:
        return self.data.__getitem__(index)
