import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

class TripletLossCosineSimilarity(nn.Module):
    def __init__(self, margin=0.8):
        super(TripletLossCosineSimilarity, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_similarity = self.cosine_similarity(anchor, positive)
        neg_similarity = self.cosine_similarity(anchor, negative)
        loss = F.relu(neg_similarity - pos_similarity + self.margin)
        return loss.mean()

class TripletLossEuclideanDistance(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossEuclideanDistance, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

class SiameseNetwork(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 distance_function: nn.Module = nn.PairwiseDistance(p=2),
                 dataset: 'torch.utils.data.Dataset' = None,
                 device ='cpu') -> None:
        
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.distance_function = distance_function
        self.device = device

        if dataset is not None:
            self.support_set_embeddings, self.labels = self._get_embeddings(dataset)
        else:
            self.support_set_embeddings = None 
            self.labels = None   

    @staticmethod
    def select_images_support_set(dataset, random=False):
        # not working
        """
        Select one image per class randomly from the dataset.
        """
        class_images = {}

        # randomly shuffle the dataset
        if random:
            dataset_indices = list(range(len(dataset)))
            random.shuffle(dataset_indices)
        else:
            dataset_indices = range(len(dataset))
        
        # Select one image per class
        for idx in dataset_indices:
            image, label = dataset[idx]
            if label not in class_images:
                class_images[label] = image
            # Stop if we have one image for each class
            if len(class_images) == len(dataset.classes):
                break

        # Get all the unique class images
        images = list(class_images.values())
        labels = list(class_images.keys())
        
        return images, labels     

    def _get_embeddings(self, dataset, random=False):
        # not working
        support_imgs, labels = SiameseNetwork.select_images_support_set(dataset, random=random)

        support_embeddings = []
        with torch.no_grad():
            with open('siamese/support_images_n.txt', 'w') as f:
                for img in support_imgs:
                    img = img.unsqueeze(0).to(self.device)  # Unsqueeze and move to device
                    f.write(f"{img} \n\n")
                    embedding = self.forward_once(img)
                    f.write(f"{embedding} \n\n")
                    support_embeddings.append(embedding)

        return support_embeddings, labels
    
    def predict(self, x):
        # not working
        if self.support_set_embeddings is None:
            raise ValueError("Support set embeddings are not available. Please provide a dataset to the constructor.")

        x = x.to(self.device)
        x_embedding = self.forward_once(x)

        distances = []
        with open('siamese/embeddings_n.txt', 'a') as f:
            for embedding in self.support_set_embeddings:
                f.write(f"{embedding} \n\n")
                distances.append(self.distance_function(x_embedding, embedding).item())
            
            f.write("x_embedding: \n")
            f.write(f"{x_embedding} \n\n")

        for i, distance in enumerate(distances):
            print(f"{i}: {distance}")

        min_distance_index = distances.index(min(distances))
        predicted_label = self.labels[min_distance_index]

        return predicted_label

    def forward_once(self, x):
        return self.backbone(x)
    
    def triplet_forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output
