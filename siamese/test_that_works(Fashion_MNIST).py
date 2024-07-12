import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
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

def select_images_support_set(dataset, random=False):
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

# Get images and compute embeddings
support_imgs, labels = select_images_support_set(fashion_MNIST_test_dataset)
support_embeddings = []

with torch.no_grad():
    with open('siamese/support_images.txt', 'w') as f:
        for img in support_imgs:
            img = img.unsqueeze(0).to(device)  # Unsqueeze and move to device
            f.write(f"{img} \n\n")
            embedding = siamese_network.forward_once(img)
            f.write(f"{embedding} \n\n")
            support_embeddings.append(embedding)

correct, incorrect = 0, 0
true_labels, predicted_labels = [], []

for idx, (x, label) in enumerate(data_loader):
    x = x.to(device)
    x_embedding = siamese_network.forward_once(x)

    distances = []
    euclidean_distance = nn.PairwiseDistance(p=2)
    with open('siamese/test_images/embeddings.txt', 'w') as f:
        for embedding in support_embeddings:
            f.write(f"{embedding} \n\n")
            distances.append(euclidean_distance(x_embedding, embedding).item())
        
        f.write("x_embedding: \n")
        f.write(f"{x_embedding} \n\n")

    min_distance_index = distances.index(min(distances))
    predicted_label = labels[min_distance_index]
    
    if predicted_label == label.item():
        correct +=1
    else: 
        incorrect +=1
    
    true_labels.append(label.item())
    predicted_labels.append(predicted_label)
    break

"""print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Plot confusion matrix with values
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.plasma)

# Add annotations with values
for i in range(len(fashion_MNIST_test_dataset.classes)):
    for j in range(len(fashion_MNIST_test_dataset.classes)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(fashion_MNIST_test_dataset.classes))
plt.xticks(tick_marks, fashion_MNIST_test_dataset.classes, rotation=45)
plt.yticks(tick_marks, fashion_MNIST_test_dataset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('siamese/test_images/confusion_matrix.png')
 """