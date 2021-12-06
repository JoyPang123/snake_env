import torch.nn as nn
from torch.distributions import Categorical


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
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
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

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy