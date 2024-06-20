import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from siamese_network import SiameseNetwork
from siamese_fashion_MNIST_backbone_network import SiameseFashionMNISTBackboneNetwork
from classifier_dataset import ClassifierDataset
from torchvision.datasets import FashionMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean, std = 0.28604059698879553, 0.35302424451492237
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Create Omniglot dataset and dataloader
fashion_MNIST_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
triplet_dataset = ClassifierDataset(fashion_MNIST_dataset)
dataloader = DataLoader(triplet_dataset, batch_size=256, shuffle=True)

# Initialize the Siamese network and move it to the device
backbone_network = SiameseFashionMNISTBackboneNetwork().to(device)
siamese_network = SiameseNetwork(backbone_network, training_classifier=True, in_features=512).to(device)

siamese_network.load_state_dict(torch.load("siamese/backbone.pth"))

# Correct loss function initialization
criterion = nn.BCELoss()

# Initialize the optimizer with a different variable name
optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)

# Initialize the accuracy variable
total_correct = 0
total_samples = 0

# Training loop
print('Starting training...')
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (img1, img2, label) in enumerate(dataloader):
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        output = siamese_network.classifier_forward(img1, img2)
        loss = criterion(output, label.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = output.round()  
        correct = (predictions == label).sum().item()
        total_correct += correct
        total_samples += label.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss/(batch_idx+1):.4f}, Accuracy: {total_correct/total_samples:.4f}')

    epoch_loss = total_loss / len(dataloader)
    epoch_accuracy = total_correct / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

torch.save(siamese_network.state_dict(), 'siamese_fashion_MNIST_model_classifier.pth')

