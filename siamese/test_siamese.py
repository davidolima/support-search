import torch
import random
from siamese_network import SiameseNetwork
from siamese_fashion_MNIST_backbone_network import SiameseFashionMNISTBackboneNetwork
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# support set
mean, std = 0.28604059698879553, 0.35302424451492237
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Create fashioMNIST dataset and dataloader
fashion_MNIST_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Assuming test_dataset is already loaded FashionMNIST test dataset
support_set_size = 1  # Number of support images per class
support_set_images = []
support_set_labels = []

# Collect support set images from different classes
for class_label in range(10):  # FashionMNIST has 10 classes
    class_indices = [idx for idx, label in enumerate(fashion_MNIST_dataset.targets) if label == class_label]
    random_indices = random.sample(class_indices, support_set_size)
    support_set_images.extend([fashion_MNIST_dataset[i][0] for i in random_indices])
    support_set_labels.extend([class_label] * support_set_size)

# Convert support set images and labels to tensors
support_set_images = torch.stack(support_set_images)
support_set_labels = torch.tensor(support_set_labels)
support_set = (support_set_images, support_set_labels)



# Assuming test_dataset is already loaded FashionMNIST test dataset
test_set_loader = DataLoader(fashion_MNIST_dataset, batch_size=1, shuffle=True)

# Initialize the network
backbone_network = SiameseFashionMNISTBackboneNetwork().to(device)
siamese_network = SiameseNetwork(backbone_network, support_set=support_set).to(device)

# Load the saved weights
siamese_network.load_state_dict(torch.load('siamese/siamese_MNIST.pth'))
siamese_network.eval()

true_labels = []
predicted_labels = []

for images, labels in test_set_loader:
    image = images[0]  # Get the first image from the batch
    label = labels[0].item()
    
    predicted_label = siamese_network(image.unsqueeze(0).to(device))
    
    true_labels.append(label)
    predicted_labels.append(predicted_label.item())

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot and save the confusion matrix as an image
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.savefig("siamese/confusion_matrix.png")


# Compute and print the test accuracy in percentage
accuracy = np.diag(conf_matrix).sum() / conf_matrix.sum()
accuracy_percent = accuracy * 100
print(f'Test Accuracy: {accuracy_percent:.2f}%')
