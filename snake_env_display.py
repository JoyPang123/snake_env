import gym

if __name__ == "__main__":
    env = gym.make("snake:snake-v0")
    state = env.reset()

    while True:
        env.render()
        obs, _, done, _ = env.step(env.action_space.sample())

        if done:
            env.render()
            break
