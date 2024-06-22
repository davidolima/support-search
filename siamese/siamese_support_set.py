import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random

class SiameseSupportSet(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        """
        Initializes the SiameseSupportSet with the given dataset and an optional transform.

        :param dataset: Dataset containing images and labels.
        :param transform: Optional transform to be applied to the images.
        """
        self.dataset = dataset
        self.transform = transform
        self.class_to_images = self._group_images_by_class()
        self.support_set_images, self.support_set_labels = self._prepare_support_set()

    def _group_images_by_class(self):
        """
        Groups images by their class labels.

        :return: A dictionary with class labels as keys and lists of images as values.
        """
        class_to_images = defaultdict(list)
        for idx in range(len(self.dataset)):
            image, label = self.dataset[idx]
            if self.transform:
                image = self.transform(image)
            class_to_images[label].append((image, label))
        return class_to_images

    def _prepare_support_set(self):
        """
        Prepares the support set by selecting one image per class.

        :return: A tuple containing support set images and their corresponding labels.
        """
        support_set_images = []
        support_set_labels = []
        for label, images in self.class_to_images.items():
            image, _ = random.choice(images)
            support_set_images.append(image)
            support_set_labels.append(label)
        support_set_images = torch.stack(support_set_images)
        support_set_labels = torch.tensor(support_set_labels)
        return support_set_images, support_set_labels

    def __len__(self):
        """
        Returns the number of classes in the support set.
        """
        return len(self.support_set_labels)

    def __getitem__(self, idx):
        """
        Returns the image and label at the specified index.

        :param idx: Index.
        :return: Tuple of image and label.
        """
        return self.support_set_images[idx], self.support_set_labels[idx]