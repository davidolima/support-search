import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
from siamese_fashion_MNIST_backbone_network import SiameseFashionMNISTBackboneNetwork
from siamese_network import SiameseNetwork
from torch.utils.data import DataLoader

# Define transformations and load the test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


fashion_MNIST_test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(fashion_MNIST_test_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
backbone_network = SiameseFashionMNISTBackboneNetwork().to(device)
siamese_network = SiameseNetwork(backbone_network, dataset=fashion_MNIST_test_dataset).to(device)
weight_path = 'siamese/weights/siamese_MNIST_euclidean.pth'
siamese_network.load_state_dict(torch.load(weight_path))
siamese_network.eval()

correct, incorrect = 0, 0
true_labels, predicted_labels = [], []

for idx, (x, label) in enumerate(data_loader):
    predicted_label = siamese_network.predict(x)
    
    if predicted_label == label.item():
        correct +=1
    else: 
        incorrect +=1
    
    true_labels.append(label.item())
    predicted_labels.append(predicted_label)
    break

print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")
