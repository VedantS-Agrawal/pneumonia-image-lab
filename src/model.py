import torch
from torch import nn


class TinyPneumoniaCNN(nn.Module):
    """Three convolution blocks; same parameter count at either resolution."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Linear(24, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x)).mean(dim=(2, 3))
        return self.head(x).squeeze(-1)


def image_tensor(images):
    """PneumoniaMNIST uint8 grayscale -> NCHW float32 in [-1, 1]."""
    return torch.from_numpy(images.copy()).float().unsqueeze(1) / 127.5 - 1
