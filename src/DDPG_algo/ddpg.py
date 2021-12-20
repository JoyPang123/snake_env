import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.DDPG_algo.replay_memory import ReplayMemory
from src.DDPG_algo.model import Actor, Critic


class DDPG:
    def __init__(self, memory_size, num_actions,
                 actor_lr, critic_lr, gamma,
                 tau, device, img_transforms):
        # Set up model
        self.actor = Actor(num_actions).to(device)
        self.target_actor = Actor(num_actions).to(device)
        self.target_actor.eval()
        self.critic = Critic(num_actions).to(device)
        self.target_critic = Critic(num_actions).to(device)
        self.target_critic.eval()

        # Set up optimizer and criterion
        self.critic_criterion = nn.MSELoss()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Set up transforms and other hyper-parameters
        self.device = device
        self.img_transforms = img_transforms
        self.num_actions = num_actions
        self.memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.tau = tau

    def choose_action(self, cur_state, eps):
        # Open evaluation mode
        self.actor.eval()

        # Exploration
        if np.random.uniform() < eps:
            action = np.random.randint(0, self.num_actions)
        else:  # Exploitation
            cur_state = self.img_transforms(cur_state).to(self.device).unsqueeze(0)
            action_list = self.actor(cur_state)
            action = torch.argmax(action_list, dim=-1).item()

        # Open training mode
        self.actor.train()
        return action

    def actor_update(self, batch_data):
        # Separate the data into groups
        cur_state_batch = []

        for cur_state, *_ in batch_data:
            cur_state_batch.append(self.img_transforms(cur_state).unsqueeze(0))

        cur_state_batch = torch.cat(cur_state_batch, dim=0).to(self.device)
        actor_actions = F.gumbel_softmax(torch.log(F.softmax(self.actor(cur_state_batch), dim=1)), hard=True)

        loss = -self.critic(cur_state_batch, actor_actions).mean()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_update(self, batch_data):
        # Separate the data into groups
        cur_state_batch = []
        reward_batch = []
        action_batch = []
        next_state_batch = []
        done_batch = []

        for cur_state, reward, action, next_state, done in batch_data:
            cur_state_batch.append(self.img_transforms(cur_state).unsqueeze(0))
            reward_batch.append(reward)
            action_batch.append(action)
            next_state_batch.append(self.img_transforms(next_state).unsqueeze(0))
            done_batch.append(done)

        cur_state_batch = torch.cat(cur_state_batch, dim=0).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch)
        action_batch = torch.zeros(len(batch_data), self.num_actions).scatter_(
            1, action_batch.unsqueeze(1), 1).to(self.device)
        next_state_batch = torch.cat(next_state_batch, dim=0).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)

        # Compute the TD error between eval and target
        Q_eval = self.critic(cur_state_batch, action_batch)
        next_action = F.softmax(self.target_actor(next_state_batch), dim=1)

        index = torch.argmax(next_action, dim=1).unsqueeze(1)
        next_action = torch.zeros_like(next_action).scatter_(1, index, 1).to(self.device)
        Q_target = reward_batch + self.gamma * (1 - done_batch) * self.target_critic(next_state_batch,
                                                                                     next_action).squeeze(1)

        loss = self.critic_criterion(Q_eval.squeeze(1), Q_target)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def soft_update(self):
        # EMA for both actor and critic network
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
