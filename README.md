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

![snake env](assets/snake.gif)

### Install the snake env
```shell
$ git clone https://github.com/JoyPang123/RL-Explore-with-Own-made-Env.git
$ cd RL-Explore-with-Own-made-Env
$ pip install -e snake
```

### Test the snake env
After installing the environment, users can test it using:
```shell
$ python snake_env_display.py
```

### Test the files
The test files are included in `test`. `test/envs` is for snake environment and `test/models` is for models' output dimension check.
```shell
$ python -m pytest -v test
```

### Install the dependencies for training the model
```shell
$ pip install -r requirements.txt
```

## Docker Environment
The docker environment is also provide.

### Install and run

```shell
(Terminal)
$ docker build -t snake Dockerfile
$ docker run -it --rm snake
```

### Remove the images file
```shell
$ docker rmi snake
```

## Algorithm
* DQN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoyPang123/RL-Explore-with-Own-made-Env/blob/main/src/DQN_algo/DQN.ipynb)
```shell
$ cd src/DQN_algo
```
* A2C

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoyPang123/RL-Explore-with-Own-made-Env/blob/main/src/A2C_algo/A2C.ipynb)
```shell
$ cd src/A2C_algo
```

* PPO

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoyPang123/RL-Explore-with-Own-made-Env/blob/main/src/PPO_algo/PPO.ipynb)
```shell
$ cd src/PPO_algo
```