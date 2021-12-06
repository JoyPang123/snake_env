import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """Adapted from
    https://github.com/raillab/a2c/blob/master/a2c/model.py
    """

    def __init__(self, num_actions):
        super().__init__()

        # Create the layers for the model
        self.actor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2, stride=2
            ),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1, stride=2
            ),  # (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # (2048)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Create the layers for the model
        self.critic = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32,
                kernel_size=5, padding=2, stride=2
            ),  # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64,
                kernel_size=3, padding=1, stride=2
            ),  # (64, 8, 8)
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=3, padding=1, stride=2
            ),  # (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # (2048)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        actor = F.log_softmax(self.actor(x), dim=1)
        critic = self.critic(x)

        return actor, critic
