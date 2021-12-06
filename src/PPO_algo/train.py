"""Code adapted from
https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import gym
import torchvision.transforms as transforms
from src.PPO_algo.ppo import PPO


if __name__ == "__main__":
    max_ep_len = 1000
    max_training_timesteps = int(1e5)

    print_freq = max_ep_len * 4
    log_freq = max_ep_len * 2

    update_timestep = max_ep_len * 4
    k_epochs = 40
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 3e-4
    lr_critic = 1e-3

    env = gym.make("snake:snake-v0")

    ppo_agent = PPO(env.action_space.n, lr_actor, lr_critic, gamma, k_epochs, eps_clip)

    running_reward = 0
    running_episodes = 0

    time_step = 0
    iteration = 0

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64))
    ])

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # Select action with policy
            action = ppo_agent.select_action(img_transforms(state["frame"]).unsqueeze(0))
            state, reward, done, _ = env.step(action)

            # Saving the episode information
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.done.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % print_freq == 0:
                avg_reward = running_reward / running_episodes

                print(f"Iteration:{iteration}, get average reward: {avg_reward:.2f}")

                running_reward = 0
                running_episodes = 0

            if done:
                break

        running_reward += current_ep_reward
        running_episodes += 1

        iteration += 1
