import gym

if __name__ == "__main__":
    env = gym.make("snake:snake-v0")
    state = env.reset()

    while True:
        env.render()
        obs, _, done, _ = env.step(1)

        if done:
            env.render()
            break
