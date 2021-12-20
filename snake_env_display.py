import os
import gym

if __name__ == "__main__":
    if not os.getenv('SDL_VIDEODRIVER'):
        driver = "x11"
        os.environ['SDL_VIDEODRIVER'] = driver

    env = gym.make("snake:snake-v0", render="pyglet")
    state = env.reset()

    while True:
        env.render()
        obs, _, done, _ = env.step(env.action_space.sample())

        if done:
            env.render()
            break
