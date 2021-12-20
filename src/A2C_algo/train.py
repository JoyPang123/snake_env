from collections import Counter

import torch
import torch.nn.functional as F
import gym

from src.A2C_algo.run_episode import run_episode
from src.A2C_algo.model import ActorCritic


def update_params(worker_optim, values, log_probs, rewards, device, gamma=0.9):
    logprobs = torch.cat(log_probs).float().flip(dims=(0,)).to(device)
    values = torch.cat(values).float().flip(dims=(0,)).to(device)
    rewards = torch.cat(rewards).float().flip(dims=(0,)).to(device)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

    returns = []
    ret_ = torch.tensor([0.])
    for reward in rewards:
        ret_ = reward + gamma * ret_
        returns.append(ret_)

    returns = torch.FloatTensor(returns).to(device)

    actor_loss = -1 * logprobs * (returns - values).detach()
    critic_loss = F.smooth_l1_loss(values, returns)
    loss = actor_loss.sum() + critic_loss.sum()

    worker_optim.zero_grad()
    loss.backward()
    worker_optim.step()

    return actor_loss, critic_loss, rewards.sum().item()


def worker(model, episodes, device):
    worker_env = gym.make("snake:snake-v0")
    worker_optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    max_reward = -1e9
    running_reward = 0

    for episode in range(episodes):
        total_reward = 0

        values, logprobs, rewards, actions = run_episode(worker_env, model, device)
        update_params(
            worker_optim, values, logprobs, rewards, device
        )
        total_reward += sum(rewards).item()
        running_reward += total_reward

        if (episode % 100) == 0:
            avg_reward = running_reward / (episode + 1)
            print(f"==========Episode: {episode}============")
            print(f"snake's length: {worker_env.snake.length}, average reward: {avg_reward}")

            action_counter = dict(Counter(actions))
            for key, value in action_counter.items():
                print(f"{key}:{value}", end=" ")
            print()

            if avg_reward > max_reward:
                max_reward = avg_reward
                torch.save(model.state_dict(), "best.pt")


def train(args, device, num_actions=4):
    actor_critic = ActorCritic(num_actions)
    worker(actor_critic, args["episodes"], device)
