import torch 
from torch.utils.data import Dataset
from typing import *

class SupportSet(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        n_way: int, k_shot: int,
        img_size: int , img_channels: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.n_way = n_way
        self.k_shot = k_shot
        self.img_size = img_size
        self.img_channels = img_channels
        
        self.targets = torch.concat([
            torch.full(size=(self.k_shot,), fill_value=i) for i in range(self.n_way)
        ])

    @classmethod
    def empty(
        cls,
        n_way: int,
        k_shot: int,
        img_size: int,
        img_channels: int,
    ) -> Self:
        return cls(
            data=torch.empty((n_way, k_shot, img_channels, img_size, img_size)),
            n_way=n_way,
            k_shot=k_shot,
            img_size = img_size,
            img_channels = img_channels,
        )

    @classmethod
    def from_tensor(cls, data: torch.Tensor) -> Self:
        """
        Expects tensor of dimension (N_WAY, K_SHOT, CHANNELS, WIDTH, HEIGHT)
        """
        return cls (
            data=data,
            n_way=data.size(0),
            k_shot=data.size(1),
            img_channels=data.size(2),
            img_size=data.size(3),
        )

    @classmethod
    def random_from_tensor(
        cls,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        n_way: int,
        k_shot: int,
        img_size: int,
        img_channels: int,
        device: Optional[str] = "cpu",
    ):
        data_tensor = torch.empty((n_way, k_shot, img_channels, img_size, img_size), device=device)
        assert len(train_images) == len(train_labels), f"Imbalance in the number of images ({len(train_images)}) and labels ({len(train_labels)})!"

        for i in range(n_way):
            # Get all images of class `i`
            class_imgs = train_images[torch.where(train_labels == i)]
            assert class_imgs.size(0) >= k_shot, f"Couldn't extract ({k_shot}) images from class `{i}`; found only {class_imgs.size(0)} images! "

            # Select `k` random images of current class
            k_idxs = torch.randperm(class_imgs.size(0))[:k_shot]
            class_imgs = class_imgs[k_idxs].view(k_shot,-1,img_size,img_size)
            data_tensor[i] = class_imgs

        return cls.from_tensor(data_tensor)

    def __len__(self):
        return self.data.size(0)*self.data.size(1)

    def __getitem__(self, index):
        return self.data.view(-1, self.img_channels, self.img_size, self.img_size)[index]
