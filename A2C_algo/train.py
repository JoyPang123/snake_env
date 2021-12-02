from collections import Counter

import torch
import torch.nn.functional as F
import gym

from A2C_algo.run_episode import run_episode
from A2C_algo.model import ActorCritic


def update_params(worker_optim, values, log_probs, rewards,
                  critic_coeff=1.0, gamma=0.9):
    logprobs = torch.cat(log_probs).float().flip(dims=(0,))
    values = torch.cat(values).float().flip(dims=(0,))
    rewards = torch.cat(rewards).float().flip(dims=(0,))
    # eps = np.finfo(np.float32).eps.item()

    returns = []
    ret_ = torch.tensor([0.])
    for reward in rewards:
        ret_ = reward + gamma * ret_
        returns.append(ret_)

    returns = torch.FloatTensor(returns)

    actor_loss = -1 * logprobs * (returns - values)
    critic_loss = F.smooth_l1_loss(values, returns)
    loss = actor_loss.sum() + critic_coeff * critic_loss.sum()

    worker_optim.zero_grad()
    loss.backward()
    worker_optim.step()

    return actor_loss, critic_loss, rewards.sum().item()


def worker(model, episodes):
    worker_env = gym.make("snake:snake-v0")
    worker_optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(episodes):
        total_reward = 0

        values, logprobs, rewards, actions = run_episode(worker_env, model)
        actor_loss, critic_loss, get_reward = update_params(
            worker_optim, values, logprobs, rewards
        )
        total_reward += get_reward

        if (episode % 100) == 0:
            print(f"==========Episode: {episode}============")
            print(f"snake's length: {worker_env.snake.length}, reward: {total_reward}")

            action_counter = dict(Counter(actions))
            for key, value in action_counter.items():
                print(f"{key}:{value}", end=" ")
            print()


def train(args, num_actions=4):
    actor_critic = ActorCritic(num_actions)
    worker(actor_critic, args["episodes"])
