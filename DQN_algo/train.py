import argparse

import gym
import torch
import torch.nn as nn

from DQN_algo.dqn import DQN
from DQN_algo.dqn import update


def train(model, env, learning_rate,
          iteration, batch_size,
          tau=0.3, gamma=0.9):
    # Save the training info
    average_reward_history = []
    reward_history = []
    loss_history = []
    total_rewards = 0

    # Set up optimizer and criterion
    optimizer = torch.optim.Adam(
        model.eval_net.parameters(), lr=learning_rate
    )
    criterion = nn.SmoothL1Loss()

    for cur_iter in range(iteration):
        print(f"===========Iteration {cur_iter + 1}/{iteration}============")
        time_step = 0
        rewards = 0
        state = env.reset()["frame"]

        while True:
            # Choose action
            action = model.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store experience
            model.store_experience(state, reward, action, next_state["frame"], done)

            # Add rewards
            rewards += reward

            # Train if the experience is enough
            if len(model.replay) > batch_size:
                loss = update(model=model,
                              batch_size=batch_size,
                              optimizer=optimizer,
                              criterion=criterion,
                              tau=tau,
                              gamma=gamma)
                loss_history.append(loss)

            # Step into next state
            state = next_state["frame"]

            # Check whether current model is done or not
            if done:
                print(f"Iteration finished after {time_step + 1} timesteps")
                print(f"Get total rewards {rewards}")
                print(f"The length of the snake is {env.snake.length}")
                break

            time_step += 1

        reward_history.append(rewards)
        total_rewards += rewards
        average_reward_history.append(total_rewards / (cur_iter + 1))

    return average_reward_history, reward_history, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training"
    )
    parser.add_argument(
        "--iteration", type=int, default=4000,
        help="Iteration for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Gamma for reward decaying"
    )
    parser.add_argument(
        "--tau", type=float, default=0.01,
        help="Rate for replacing model"
    )
    parser.add_argument(
        "--replace_iter", type=int, default=10,
        help="Iteration for replacing target by eval network"
    )
    parser.add_argument(
        "--max_len", type=int, default=1000,
        help="Maximum len of the deque"
    )
    parser.add_argument(
        "--eps_start", type=float, default=0.5,
        help="Beginning exploration rate"
    )
    parser.add_argument(
        "--eps_end", type=float, default=-0.05,
        help="Ending exploration rate"
    )
    parser.add_argument(
        "--eps_decay", type=float, default=200,
        help="Decay rate for the exploration rate"
    )
    args = parser.parse_args()

    env = gym.make("snake:snake-v0")

    # Set up environment hyper-parameters
    num_actions = env.action_space.n

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Used: {device}")

    # Build the model
    dqn_model = DQN(
        num_actions=num_actions,
        device=device, replace_iter=args.replace_iter,
        max_len=args.max_len,
        EPS_START=args.eps_start, EPS_END=args.eps_end,
        EPS_DECAY=args.eps_decay
    )

    # Start training DQN
    average_reward_history, reward_history, loss_history = train(
        model=dqn_model, env=env, learning_rate=args.lr,
        iteration=args.iteration, batch_size=args.batch_size, tau=args.tau,
        gamma=args.gamma
    )
