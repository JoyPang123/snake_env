# README

## Snake env

### Introduction
* The snake environment supports 4 actions: **[up, right, down, top]**.
* The snake environment will have extra blocks for better learning.
* The head is denoted with blue color and the rest of the body is black.
* The apple is red.
* Reward is designed:
  * Eat apple: +20
  * Crush on walls: -10
  * Others: -((new_head - apple_pos) - (old_head - apple_pos))

![snake env](assets/output.mp4)

### Install the snake env
```shell
$ git clone https://github.com/JoyPang123/RL-Explore-with-Own-made-Env.git
$ cd RL-Explore-with-Own-made-Env
$ pip install -e snake
```

### Test the env
After installing the environment, users can test it using:
```shell
$ python snake_env_display.py
```

## Test the environment
The test files are included in `test`. `test/envs` is for snake environment and `test/models` is for models' output dimension check.
```shell
$ python -m pytest -v test
```

## Install the dependencies for training the model
```shell
$ pip install -r requirements.txt
```

