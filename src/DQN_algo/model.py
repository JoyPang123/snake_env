import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, num_actions, in_channels=3):
        super(DQN, self).__init__()

        # Create the layers for the model
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16,
                kernel_size=5, padding=2, stride=2
            ),  # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16, out_channels=32,
                kernel_size=3, padding=1, stride=2
            ),  # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1, stride=2
            ),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 4 * 4, num_actions)
        )

    def forward(self, x):
        return self.layers(x)
