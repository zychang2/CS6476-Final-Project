# Potential code, using pre-trained RN34 / RN50
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class ModResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModResNet, self).__init__()
        
        # Load the pre-trained ResNet50 model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.resnet.fc.in_features
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Replace the last fully connected layer with an identity layer
        self.resnet.fc = nn.Identity()

        # Define the fully connected layers
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Pass the input through the pre-trained ResNet34
        x = self.resnet(x)
        
        # Flatten the output of ResNet
        x = x.view(x.size(0), -1)
        
        # Pass the flattened output through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        return x