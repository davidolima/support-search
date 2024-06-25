import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import random
import datetime
from siamese_fashion_MNIST_backbone_network import SiameseFashionMNISTBackboneNetwork
from siamese_network import SiameseNetwork

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

def select_images(dataset):
    """
    Select one image per class and one additional random image from the dataset
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

    return random_image, random_label, images, labels

def plot_images(anchor, images, distances, labels):
    """
    Plot the images with their distances and highlight the matching class image
    """
    anchor = anchor.cpu().squeeze().numpy()
    fig, axes = plt.subplots(1, len(images) + 1, figsize=(20, 4))
    axes[0].imshow(anchor, cmap='gray')
    axes[0].set_title(f'Label: {labels[0]}\nAnchor Image')
    axes[0].axis('off')

    # Find the index of the image with the smallest distance
    min_distance_index = distances.index(min(distances))

    for i, (img, dist, label) in enumerate(zip(images, distances, labels[1:])):
        img = img.cpu().squeeze().numpy()
        axes[i+1].imshow(img, cmap='gray')
        if label == labels[0]:  # Highlight the matching class image
            axes[i+1].imshow(img, cmap='gray', alpha=0.7)
            rect = plt.Rectangle((0, 0), 27, 27, linewidth=5, edgecolor='r', facecolor='none')
            axes[i+1].add_patch(rect)
        if i == min_distance_index:  # Highlight the image with the smallest distance
            rect = plt.Rectangle((0, 0), 27, 27, linewidth=3, edgecolor='b', facecolor='none')
            axes[i+1].add_patch(rect)
        axes[i+1].set_title(f'Label: {label}\nDistance: {dist:.2f}')
        axes[i+1].axis('off')

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'siamese/test_images/triplet_images_{timestamp}.png')

# Get images and compute embeddings
anchor_img, anchor_label, support_imgs, labels = select_images(fashion_MNIST_test_dataset)
support_embeddings = []
with torch.no_grad():
    anchor_embedding = siamese_network.forward_once(anchor_img)
    for img in support_imgs:
        img = img.unsqueeze(0).to(device)  # Unsqueeze and move to device
        embedding = siamese_network.forward_once(img)
        support_embeddings.append(embedding)

# Calculate euclidean distances
distances = []
euclidean_distance = nn.PairwiseDistance(p=2)
for embedding in support_embeddings:
    distances.append(euclidean_distance(anchor_embedding, embedding).item())

# Plot the results
plot_images(anchor_img, support_imgs, distances, [anchor_label] + labels)
