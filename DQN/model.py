import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super(Model, self).__init__()

        # Create the layers for the model
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16,
                kernel_size=5, padding=2, stride=2
            ),  # (16, 16, 16)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16, out_channels=8,
                kernel_size=5, padding=2, stride=2
            ),  # (8, 8, 8)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 8, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
