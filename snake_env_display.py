import gym
import cv2

if __name__ == "__main__":
    env = gym.make('snake:snake-v0')
    state = env.reset()

    # Save the result
    cv2.imwrite("test.jpg", cv2.cvtColor(state["frame"], cv2.COLOR_RGB2BGR))

    while True:
        env.render()
        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)

        if done:
            env.render()
            break
