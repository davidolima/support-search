from torch.utils.data import Dataset
import random

class ClassifierDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.targets = dataset.targets.tolist()  # Convert to list if it's a tensor
        self.classes = list(set(self.targets))  # Extract unique class labels
        # Create a dictionary mapping each class to the list of indices of its samples
        self.class_to_indices = {cls: [] for cls in self.classes}
        for idx, label in enumerate(self.targets):
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get the first image and its label
        img1, label1 = self.dataset[index]

        # Determine if the second image should be from the same class or a different class
        if random.random() < 0.5:  # 50% chance of being from the same class
            cls = label1
            img2_index = index
            # Ensure img2_index is not the same as index to avoid duplication
            while img2_index == index:
                img2_index = random.choice(self.class_to_indices[cls])
            img2, label2 = self.dataset[img2_index]
            label = 1  # Same class label
        else:
            cls = random.choice([c for c in self.classes if c != label1])
            img2_index = random.choice(self.class_to_indices[cls])
            img2, label2 = self.dataset[img2_index]
            label = 0  # Different class label

        return img1, img2, label

