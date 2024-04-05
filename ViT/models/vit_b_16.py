import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights

# Define the model
class ModViT(nn.Module):
    def __init__(self, num_classes=2):
        super(ModViT, self).__init__()
        
        # Load the pre-trained ViT model
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # First try to freeze the transf
        for param in self.vit.parameters():
            param.requires_grad = True
        num_ftrs = self.vit.heads.head.out_features

        # Define the fully connected layers
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Pass the input through the pre-trained ViT
        x = self.vit(x)
        
        # Flatten the output of ViT
        x = x.view(x.size(0), -1)
        
        # Pass the flattened output through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        return x



import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class Attention(nn.Module):
    def __init__(self, feature_dim: int):
        super(Attention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        
        # Apply softmax to get probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # Weighted sum of values
        attention_output = torch.matmul(attention_probs, value)
        return attention_output

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
        sparsity_weight: float = 0.2,
        sparsity_target: float = 0.1,
    ):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            Attention(hidden_dim1), # Attention layer
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, output_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid(),
        )

        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        sparsity_loss = torch.mean(torch.abs(encoded - self.sparsity_target))
        return encoded, decoded, sparsity_loss * self.sparsity_weight