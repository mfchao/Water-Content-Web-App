import torch
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.feature_extractor.fc = nn.Linear(512, 128)
        self.regression_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.regression_head(x)
        return x