import torch
import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self, name='resnet_10', out_dim=128, apply_bn=False, device="cuda"):
        super(Projector, self).__init__()
        _, dim_in = model_dict[name]
        self.linear1 = nn.Linear(dim_in, dim_in)
        self.linear2 = nn.Linear(dim_in, out_dim)
        self.bn = nn.BatchNorm1d(dim_in)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)
        self.projector = self.projector.to(device)

    def forward(self, x):
        return self.projector(x)

class LinearClassifier(nn.Module):
    def __init__(self, name='resnet_10', num_classes=2, device="cuda"):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes).to(device)

    def forward(self, features):
        return self.fc(features)

model_dict = {
    'resnet_10' : ['resnet_10', 512]
}