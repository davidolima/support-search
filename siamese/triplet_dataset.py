from torch.utils.data import Dataset
import random

class TripletDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.labels = [label for _, label in self.dataset]
        self.label_to_indices = {label: [idx for idx, l in enumerate(self.labels) if l == label] for label in set(self.labels)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor, anchor_label = self.dataset[idx]
        
        # Positive sample
        positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive, _ = self.dataset[positive_idx]
        
        # Negative sample
        negative_label = random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative, _ = self.dataset[negative_idx]

        return anchor, positive, negative