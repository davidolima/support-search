import torch
import torch.nn as nn
from typing import Optional

class L1Distance(nn.Module):
    def __init__(self):
        super(L1Distance, self).__init__()

    def forward(self, x1, x2):
        return torch.abs(x1 - x2).sum(dim=1, keepdim=True)

class SiameseNetwork(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 training_classifier: bool = False,
                 in_features: int = 4096, 
                 distance_function: nn.Module = None,
                 support_set: Optional[nn.Module] = None) -> None:
        
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.distance_function = distance_function if distance_function is not None else L1Distance()

        if support_set is not None:
            self.support_set = support_set
            self.n_classes = support_set.n_way

        if training_classifier:
            # Freeze backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        return self.backbone(x)
    
    def triplet_foward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output

    def classifier_forward(self, x1, x2):
        print(f"img1: {x1.shape} img2: {x2.shape}")
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        distances = self.distance_function(output1, output2)

        return self.classifier(distances)


    def forward(self, x):
        if self.support_set is None:
            raise RuntimeError("Support set not provided.")

        similarities = []
        for support_image, _ in self.support_set:
            output1 = self.forward_once(x)
            output2 = self.forward_once(support_image)
            distance = self.distance_function(output1, output2)
            similarity = self.classifier(distance)
            similarities.append(similarity.item())
        
        # Find the index of the support image with the highest similarity
        predicted_index = similarities.index(max(similarities))
        predicted_class = self.support_set[predicted_index][1]
        return predicted_class

