import torch
from torch import nn

class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.features = self._make_feature_extractor()
        self.fc = self._make_classifier()

    def _make_feature_extractor(self):
        layers = [
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),  # Dropout layer added
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1)   # Dropout layer added
        ]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = [
            nn.Linear(64 * 13 * 13, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer added
            nn.Linear(256, 1),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = torch.abs(output1 - output2)
        return distance  # Ensure this returns a shape of (batch_size, 1)
