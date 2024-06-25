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


def generate_support_set(dataset):
    """
    Select one image per class.
    """
    class_images = {}
    support_set = []
    selected_labels = set()
    
    # Select one image per class
    for image, label in dataset:
        if label not in selected_labels:
            class_images[label] = image
            selected_labels.add(label)
        # Stop if we have one image for each class
        if len(selected_labels) == len(dataset.classes):
            break
    
    # Create the support set
    support_set = [(image, label) for label, image in class_images.items()]

    return support_set


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
backbone_network = SiameseFashionMNISTBackboneNetwork().to(device)
support_set = generate_support_set(fashion_MNIST_test_dataset)
siamese_network = SiameseNetwork(backbone_network, support_set=support_set).to(device)

# Load the weights
weight_path = 'siamese/weights/siamese_MNIST_euclidean.pth'
siamese_network.load_state_dict(torch.load(weight_path))
siamese_network.eval()

data_loader = DataLoader(fashion_MNIST_test_dataset, batch_size=1, shuffle=True)

for image, labels in data_loader:
    # Move images to the device (GPU if available)
    image = image.to(device)
    predicted_label = siamese_network(image)
    # Plot each image in the batch
    for i in range(image.size(0)):
        image = image[i].squeeze().cpu().numpy()  # Convert to numpy array
        label = labels[i].item()  # Get the label as an integer
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')
        plt.savefig('siamese/test_images/x_image.png')

    
    print(f"Predicted label: {predicted_label}, True label: {label}")
    break

