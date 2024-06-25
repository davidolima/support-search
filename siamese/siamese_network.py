import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
                 support_set: any = None,
                 device = 'cpu') -> None:
        
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.distance_function = distance_function
        self.device = device

        if support_set is not None:
            self.support_set_embeddings, self.labels = self._get_embeddings(support_set)
        else:
            self.support_set_embeddings = None 
            self.labels = None           

    def _get_embeddings(self, support_set):
        # not working
        support_set_embeddings = []
        labels = []
        for image, label in support_set:
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.forward_once(image)
            
            support_set_embeddings.append(embedding)
            labels.append(label)

        return support_set_embeddings, labels
    
    def forward_once(self, x):
        return self.backbone(x)
    
    def triplet_foward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output
    
    def forward(self, x):
        # Foward does not work
        if self.support_set_embeddings is None:
           raise ValueError("Support set was not provided")
        
        distances = []
        for embedding in self.support_set_embeddings:
            distances.append(self.distance_function(self.forward_once(x), embedding).item())

        # Calculate accuracy
        min_distance_index = distances.index(min(distances))
        predicted_label = self.labels[min_distance_index]

        return predicted_label