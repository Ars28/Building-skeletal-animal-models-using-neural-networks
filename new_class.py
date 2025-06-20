import torch.nn as nn
import torchvision
from torch import stack
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        # Бэкбон
        self.backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()

        # Головы
        self.fc_shared = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_x = nn.Linear(512, num_keypoints // 2)
        self.fc_y = nn.Linear(512, num_keypoints // 2)

        # Инициализация весов голов
        for m in [self.fc_x, self.fc_y]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        features = self.backbone(x)
        shared = self.fc_shared(features)
        x_coords = self.fc_x(shared)
        y_coords = self.fc_y(shared)
        return stack([x_coords, y_coords], dim=2).flatten(1)