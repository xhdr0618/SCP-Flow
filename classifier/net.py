'''
image classifier

The main components include:
1. A ClassifierHead class that handles the final classification layers with customizable hidden dimensions
2. A Classifier class that combines the ResNet50 backbone with the classifier head
3. Utility functions for feature extraction, prediction, and model loading/saving
'''

import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import Optional, Tuple, Union, List
import numpy as np
import os

class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int = 2, hidden_dim: Union[Tuple[int], int] = 256,
                 dropout_rate: float = 0.5, use_layer_norm: bool = True):
        super().__init__()

        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        layers = []
        prev_dim = in_features

        self.norm = nn.LayerNorm(in_features) if use_layer_norm else nn.BatchNorm1d(in_features)

        for dim in hidden_dim:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.classifier(x)


class Classifier(nn.Module):
    def __init__(
            self,
            hidden_dim: tuple =(512,512),
            n_classes: int = 2,
            dropout_rate: float = 0.5,
            device="cuda",
            latent_size=(2048, 1, 1),
            freeze_backbone=False,
    ):
        """
        Binary classifier with modular backbone and classification head
        Args:
            hidden_dim: dimensions of hidden layers in classifier head
            dropout_rate: dropout rate for classifier head
            freeze_backbone: whether to freeze backbone parameters
        """
        super().__init__()
        self.device = device
        # Initialize backbone (ResNet50)
        self.backbone = resnet50(pretrained=True)

        # Remove original FC layer and keep only the feature extractor
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Initialize classifier head as a separate module
        self.cls_head = ClassifierHead(
            in_features=int(np.prod(latent_size)),
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using backbone only
        Args:
            x: input image tensor of shape (batch_size, 3, 256, 256)
        Returns:
            feature tensor
        """
        features = self.backbone(x)
        return features.view(features.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through entire model
        Args:
            x: input image tensor of shape (batch_size, 3, 256, 256)
        Returns:
            classification logits of shape (batch_size, 2)
        """
        # Extract features using backbone
        features = self.extract_features(x)

        # Pass through classifier head
        logits = self.cls_head(features)
        return logits

    def predict_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Prediction interface
        Args:
            feature: input pred vqgan tensor of shape (batch_size, 4, 32, 32)
        Returns:
            predicted logit of shape (batch_size, num_class)
        """
        self.eval()
        with torch.no_grad():
            logits = self.cls_head(feature.view(feature.size(0), -1))
        return logits
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prediction interface
        Args:
            x: input image tensor of shape (batch_size, 3, 256, 256)
        Returns:
            predicted class indices of shape (batch_size, num_class)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
        return logits


def load_classifier(
        ckpt_path: str,
        hidden_dim: tuple = (512,512),
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Classifier:
    """
    Load trained classifier
    Args:
        ckpt_path: path to checkpoint file
        hidden_dim: hidden layer dimensions, use default if None
        device: device to load model on
    Returns:
        loaded classifier model
    """
    model = Classifier(hidden_dim=hidden_dim, n_classes=2) if hidden_dim else Classifier()
    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    """ start - fixing some ugly bugs  """
    # Check if the keys in state_dict contain the "model." prefix
    has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())

    if has_model_prefix:
        # If there is a "model." prefix, create a new state_dict removing the prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('model.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict
    """ end - fixing some ugly bugs  """

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# Usage example
def example_usage():
    # Create model
    model = Classifier(hidden_dim=(512,512)).to("cuda")

    # Example: Using backbone only for feature extraction
    sample_input = torch.randn(1, 3, 256, 256).to("cuda")
    features = model.extract_features(sample_input)
    print(f"Extracted features shape: {features.shape}")

    # Example: Full classification
    logits = model(sample_input)
    print(f"Classification logits shape: {logits.shape}")

    # Save model
    checkpoint = {
        'state_dict': model.state_dict(),
        'hidden_dim': 128,
    }
    torch.save(checkpoint, 'classifier.pth')

    # Load model
    loaded_model = load_classifier('classifier.pth')
    prediction = loaded_model.predict(sample_input)
    print(f"Prediction shape: {prediction.shape}")


if __name__ == '__main__':
    os.environ['TORCH_HOME'] = "../pre-trained"  # path to save torch pre-train ckpt
    example_usage()