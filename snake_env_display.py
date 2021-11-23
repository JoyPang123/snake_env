import gym

if __name__ == "__main__":
    env = gym.make('snake:snake-v0')
    state = env.reset()

    while True:
        env.render()
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        print(reward)

        if done:
            env.render()
            break
