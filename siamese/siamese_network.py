import torch.nn as nn
import torch.nn.functional as F


class TripletLossCosineSimilarity(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLossCosineSimilarity, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        pos_similarity = self.cosine_similarity(anchor, positive)
        neg_similarity = self.cosine_similarity(anchor, negative)
        loss = F.relu(neg_similarity - pos_similarity + self.margin)
        return loss.mean()
    
class TripletLossEuclideanDistance(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLossEuclideanDistance, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = (anchor - positive).pow(2).sum(1)
        neg_distance = (anchor - negative).pow(2).sum(1)
        loss = F.relu(neg_distance - pos_distance + self.margin)
        return loss.mean()

class SiameseNetwork(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 distance_function: nn.Module = nn.CosineSimilarity(dim=1),
                 support_set: any = None) -> None:
        
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.distance_function = distance_function

        if support_set is not None:
            self.support_set_embeddings = self._get_embeddings(support_set)
        else:
            self.support_set_embeddings = None            

    def _get_embeddings(self, support_set):
        # 1 way k shot
        support_set_images, support_set_labels = support_set
        embeddings = self.forward_once(support_set_images)
        print("===============================")
        print(embeddings)
        print(support_set_labels.shape)
        print(support_set_labels)
        print("===============================")
        return list(zip(embeddings, support_set_labels))
           
    def forward_once(self, x):
        return self.backbone(x)
    
    def triplet_foward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output
    
    def forward(self, x):
        if self.support_set_embeddings is None:
            raise RuntimeError("Support set not provided.")
        
        similarities = []
        x_embedding = self.forward_once(x)
        for support_embedding, label in self.support_set_embeddings:
            #print("=====================================")
            #print(support_embedding)
            #print(x_embedding)
            
            distance = self.distance_function(x_embedding, support_embedding)
            #print(distance)
            #print("=====================================")
            similarities.append((distance.item(), label))
        
        # Find the label of the support image with the highest similarity
        predicted_label = max(similarities, key=lambda item: item[0])[1]
        return predicted_label

