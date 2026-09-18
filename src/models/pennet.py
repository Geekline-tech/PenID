import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PenNet(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class PenNetWithClassifier(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 40, pretrained: bool = True):
        super().__init__()
        self.pennet = PenNet(embed_dim, pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.pennet(x)
        logits = self.classifier(emb)
        return emb, logits
