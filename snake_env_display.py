import gym

env = gym.make('snake:snake-v0')
state = env.reset()

while True:
    env.render()
    action = env.action_space.sample()
    _, _, done, _ = env.step(action)

    if done:
        break
