import math
import random
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from src.DQN_algo.model import DQN


def update(model, batch_size,
           optimizer, criterion,
           tau=0.3,
           gamma=0.9):
    # Set up the device same as model
    used_device = model.device
    # Get the data from the experience
    batch_data = random.sample(model.replay,
                               batch_size)

    # Separate the data into groups
    cur_state_batch = []
    reward_batch = []
    action_batch = []
    next_state_batch = []
    done_batch = []

    for cur_state, reward, action, next_state, done in batch_data:
        cur_state_batch.append(model.transforms(cur_state).unsqueeze(0))
        reward_batch.append(reward)
        action_batch.append(action)
        next_state_batch.append(model.transforms(next_state).unsqueeze(0))
        done_batch.append(done)

    cur_state_batch = torch.cat(cur_state_batch, dim=0).to(used_device)
    reward_batch = torch.FloatTensor(reward_batch).to(used_device)
    action_batch = torch.FloatTensor(action_batch).to(used_device)
    next_state_batch = torch.cat(next_state_batch, dim=0).to(used_device)
    done_batch = torch.Tensor(done_batch).to(used_device)

    # Compute the error between eval and target net
    Q_eval = model.eval_net(cur_state_batch).gather(
        dim=1,
        index=action_batch.long().unsqueeze(1)
    ).squeeze(1)

    # Detach from target net to avoid computing the gradient
    Q_next = model.target_net(next_state_batch).detach()
    Q_target = reward_batch + gamma * (1 - done_batch) * torch.max(Q_next, dim=1)[0]

    # Compute loss and update the model
    loss = criterion(Q_eval, Q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Add the counter for the eval
    model.step_counter += 1

    # Replace target net with eval net
    if model.step_counter == model.replace_iter:
        model.step_counter = 0
        for eval_parameters, target_parameters in zip(model.eval_net.parameters(),
                                                      model.target_net.parameters()):
            target_parameters.data.copy_(tau * eval_parameters.data + \
                                         (1.0 - tau) * target_parameters.data)

    return loss.item()


class DQN:
    def __init__(self, num_actions, device,
                 replace_iter=150, max_len=100,
                 EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        # Create network for target and evaluation
        self.eval_net = DQN(num_actions=num_actions).to(device)
        self.target_net = DQN(num_actions=num_actions).to(device)

        # Set up the replay experience
        self.replay = deque(maxlen=max_len)

        # Transform the image
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (32, 32), interpolation=InterpolationMode.BICUBIC
            )
        ])

        # Set up the counter to update target from eval
        self.target_counter = 0

        # Set up hyper-parameters
        self.device = device
        self.num_actions = num_actions
        self.replace_iter = replace_iter
        self.step_counter = 0

        # For exploration probability
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.step_total_count = 0

    def choose_action(self, cur_state):
        # Open evaluation mode
        self.eval_net.eval()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) \
                        * math.exp(-1. * self.step_total_count / self.EPS_DECAY)
        self.step_total_count += 1

        """Choose the action using epsilon greedy policy"""
        # Exploration
        if np.random.uniform() < eps_threshold:
            action = np.random.randint(0, self.num_actions)
        else:  # Exploitation
            cur_state = self.transforms(cur_state).to(self.device).unsqueeze(0)
            action_list = self.eval_net(cur_state)
            action = torch.argmax(action_list, dim=-1).item()

        # Open training mode
        self.eval_net.train()
        return action

    def store_experience(self, state, reward,
                         action, next_state,
                         done):
        """Record the play experience into deque

        The format of the experience:
            [state, reward, action, next_state, done]
        """

        self.replay.append([state, reward, action, next_state, done])
