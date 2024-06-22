import torch.nn as nn
import torch.nn.functional as F
import torch

class SiameseFashionMNISTBackboneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, 5)  
        self.conv2 = nn.Conv2d(64, 128, 3)  
        self.fc1 = nn.Linear(128 * 5 * 5, 512)  # Adjusted the output size of fc1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First block of convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # Second block of convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer with sigmoid activation
        x = torch.sigmoid(self.fc1(x))
        return x
    
if __name__ == '__main__':
    # Create a random tensor with the shape of (1, 1, 28, 28)
    x = torch.randn(1, 1, 28, 28)
    
    # Initialize the SiameseFashionMNISTBackboneNetwork
    model = SiameseFashionMNISTBackboneNetwork()
    
    # Forward pass the random tensor
    output = model(x)