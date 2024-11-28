import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.base_model = models.resnet50(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x)
        # x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)
        return anchor_embedding, positive_embedding, negative_embedding
