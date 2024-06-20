import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseOmniglotBackboneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First block of convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Second block of convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Third block of convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # Fourth block of convolutional layer with ReLU activation
        x = F.relu(self.conv4(x))
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer with sigmoid activation
        x = torch.sigmoid(self.fc1(x))

        return x


if __name__ == '__main__':
    net = SiameseOmniglotBackboneNetwork()
    x = torch.randn(1, 1, 105, 105)
    net(x)