
import torch.nn as nn
from torchvision import models


class PlantDiseaseModel(nn.Module):
    def __init__(self, architecture="mobilenet_v2", num_classes=15):
        super().__init__()

        if architecture == "mobilenet_v2":
            self.model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT
            )

            # Freeze pretrained feature extractor
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Replace classification layer
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(
                in_features, num_classes
            )

        elif architecture == "efficientnet_b0":
            self.model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT
            )

            # Freeze pretrained feature extractor
            for param in self.model.features.parameters():
                param.requires_grad = False

            # Replace classification layer
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(
                in_features, num_classes
            )

        else:
            raise ValueError(
                f"Unsupported architecture: {architecture}"
            )

    def forward(self, x):
        return self.model(x)

