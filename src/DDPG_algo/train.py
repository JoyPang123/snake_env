import torch
import torchvision.transforms as transforms

import gym

from src.DDPG_algo.ddpg import DDPG


def train(max_time_steps, max_iter, memory_size,
          num_actions, actor_lr, critic_lr,
          gamma, tau, device, batch_size):
    env = gym.make("snake:snake-v0", mode="hardworking")

    # Set up model training
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    ddpg = DDPG(
        memory_size, num_actions,
        actor_lr, critic_lr, gamma,
        tau, device, img_transforms
    )
    max_reward = 0

    running_reward = 0
    running_episodes = 0

    time_step = 0
    print_freq = max_iter * 2

    while time_step < max_time_steps:
        state = env.reset()
        current_ep_reward = 0

        for _ in range(max_iter):
            # Get reward and state
            actions = ddpg.choose_action(state["frame"], 0.1)
            new_state, reward, done, _ = env.step(actions)

            current_ep_reward += reward
            ddpg.memory.store_experience(state["frame"], reward, actions, new_state["frame"], done)
            state = new_state

            if done:
                break

            # Wait for updating
            if ddpg.memory.size() < batch_size:
                continue
                
            batch_data = ddpg.memory.sample(batch_size)
            ddpg.critic_update(batch_data)
            ddpg.actor_update(batch_data)
            ddpg.soft_update()

            time_step += 1

            if time_step % print_freq == 0:
                avg_reward = running_reward / running_episodes

                print(f"Iteration:{running_episodes}, get average reward: {avg_reward:.2f}")

                running_reward = 0
                running_episodes = 0

                if avg_reward > max_reward:
                    max_reward = avg_reward
                    torch.save(ddpg.actor.state_dict(), "actor_best.pt")
                    torch.save(ddpg.critic.state_dict(), "critic_best.pt")

        running_reward += current_ep_reward
        running_episodes += 1
