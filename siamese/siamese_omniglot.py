import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from siamese_network import SiameseNetwork
from siamese_omniglot_backbone_network import SiameseOmniglotBackboneNetwork
from triplet_dataset import TripletDataset

from torchvision.datasets import Omniglot


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create Omniglot dataset and dataloader
omniglot_dataset = Omniglot(root='./data', download=True, transform=transform)
triplet_dataset = TripletDataset(omniglot_dataset)
dataloader = DataLoader(triplet_dataset, batch_size=64, shuffle=True)

# Initialize the Siamese network and move it to the device
backbone_network = SiameseOmniglotBackboneNetwork().to(device)
siamese_network = SiameseNetwork(backbone_network).to(device)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(siamese_network.parameters(), lr=0.001)

# Training loop
print('Starting training...')
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_embedding, positive_embedding, negative_embedding = siamese_network.triplet_foward(anchor, positive, negative)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss/(batch_idx+1):.4f}')

    print(f'Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss/len(dataloader):.4f}')

torch.save(siamese_network.state_dict(), 'siamese_omniglot_model.pth')
