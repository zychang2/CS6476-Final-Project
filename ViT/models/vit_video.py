import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from einops.layers.torch import Rearrange
from vit_pytorch import ViT

class CustomViT(nn.Module):
    def __init__(self, image_size=1200, patch_size=240, num_classes=2):
        super(CustomViT, self).__init__()
        
        assert image_size % patch_size == 0, "Image size must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 240 * 240 * 3
        
        self.patch_embedding = nn.Sequential(
            Resize((patch_size, patch_size)),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        )
        
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, patch_dim))
        
        self.vit = ViT(
            image_size=patch_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=patch_dim,
            depth=16,  # Assuming you want to keep the original depth
            heads=16,  # Assuming you want to keep the original number of heads
            mlp_dim=4096,  # Assuming you want to keep the original MLP dimension
            dropout=0.1,  # Assuming you want to keep the original dropout
            emb_dropout=0.1  # Assuming you want to keep the original embedding dropout
        )
        
    def forward(self, x):
        x = self.patch_embedding(x)
        x += self.positional_encoding
        x = self.vit(x)
        return x