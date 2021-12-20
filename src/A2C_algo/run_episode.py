import torch
from torch.distributions import Categorical

import torchvision.transforms as transforms


def run_episode(worker_env, worker_model, device, n_steps=1000):
    # Transform the image
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    state = worker_env.reset()
    state = img_transforms(state["frame"])
    values, logprobs, rewards, actions = [], [], [], []

    count_length = 0

    while True:
        count_length += 1
        policy, value = worker_model(state.unsqueeze(0).float().to(device))

        values.append(value.view(-1))
        logits = policy.view(-1)
        action_dist = Categorical(logits=logits)
        action = action_dist.sample()
        actions.append(action.item())

        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_.view(-1))

        state_, reward, done, info = worker_env.step(action.item())
        state = img_transforms(state_["frame"])

        rewards.append(torch.tensor([reward]))
        if done:
            break

    return values, logprobs, rewards, actions
