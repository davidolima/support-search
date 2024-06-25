import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
from siamese_fashion_MNIST_backbone_network import SiameseFashionMNISTBackboneNetwork
from siamese_network import SiameseNetwork
from torch.utils.data import DataLoader
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
backbone_network = SiameseFashionMNISTBackboneNetwork().to(device)
siamese_network = SiameseNetwork(backbone_network).to(device)
weight_path = 'siamese/weights/siamese_MNIST_euclidean.pth'
siamese_network.load_state_dict(torch.load(weight_path))
siamese_network.eval()

# Define transformations and load the test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

fashion_MNIST_test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(fashion_MNIST_test_dataset, batch_size=1, shuffle=True)

def select_images_support_set(dataset):
    """
    Select one image per class 
    """
    class_images = {}
    
    # Select one image per class
    for image, label in dataset:
        if label not in class_images:
            class_images[label] = image
        # Stop if we have one image for each class
        if len(class_images) == len(dataset.classes):
            break

    # Get all the unique class images
    images = list(class_images.values())
    labels = list(class_images.keys())
    
    # Add one random image from the dataset
    random_image, random_label = random.choice(dataset)
    random_image = random_image.unsqueeze(0).to(device)

    return images, labels

# Get images and compute embeddings
support_imgs, labels = select_images_support_set(fashion_MNIST_test_dataset)
support_embeddings = []

with torch.no_grad():
    for img in support_imgs:
        img = img.unsqueeze(0).to(device)  # Unsqueeze and move to device
        embedding = siamese_network.forward_once(img)
        support_embeddings.append(embedding)

correct, incorrect = 0, 0

for idx, (x, label) in enumerate(data_loader):
    x_embedding = siamese_network.forward_once(x)

    distances = []
    euclidean_distance = nn.PairwiseDistance(p=2)
    for embedding in support_embeddings:
        distances.append(euclidean_distance(x_embedding, embedding).item())

    min_distance_index = distances.index(min(distances))
    predicted_label = labels[min_distance_index]
    
    if predicted_label == label:
        correct +=1
    else: 
        incorrect +=1


print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")
