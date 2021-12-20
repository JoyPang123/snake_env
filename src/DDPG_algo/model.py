import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # Create the layers for the model
        self.actor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2, stride=2
            ),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1, stride=2
            ),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # (2048)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.actor(x)


class Critic(nn.Module):
    def __init__(self, act_dim):
        super().__init__()

        # Create the layers for the model
        self.critic = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2, stride=2
            ),  # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1, stride=2
            ),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # (2048)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, state, action):
        x = self.critic(state)
        x = torch.cat([x, action], dim=1)
        x = self.fc(x)

        return x
