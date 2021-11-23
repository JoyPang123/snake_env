"""Testing function adapted from
https://github.com/openai/gym/blob/master/tests/envs/test_envs.py
"""

import gym
import numpy as np
import pytest
from gym import spaces


def test_env():
    # Capture warning from making environment
    with pytest.warns(None) as warnings:
        env = gym.make("snake:snake-v0")

    # Check inherit
    assert isinstance(env, gym.Env), "Environment are not inherit from gym.Env"


def test_observation_and_action():
    # Capture warning from making environment
    with pytest.warns(None) as warnings:
        env = gym.make("snake:snake-v0")

    # Check observation and action space
    assert hasattr(env, "observation_space"), "The observation space are not specify"
    assert isinstance(env.observation_space, spaces.Space), "The observation space should be in spaces.Space"

    assert hasattr(env, "action_space"), "The action space are not specify"
    assert isinstance(env.action_space, spaces.Space), "The action space should be in spaces.Space"


def test_step_before_reset():
    # Capture warning from making environment
    with pytest.warns(None) as warnings:
        env = gym.make("snake:snake-v0")

    # Test step before reset
    try:
        env.step(env.action_space.sample())
    except AssertionError as e:
        assert str(e) == "Cannot call env.step() before calling reset()"


def test_observation_type():
    # Capture warning from making environment
    with pytest.warns(None) as warnings:
        env = gym.make("snake:snake-v0")

    # Check observation types
    obs_spaces = env.observation_space.spaces

    for key, space in obs_spaces.items():
        if isinstance(space, spaces.Box):
            assert space.dtype == np.uint8, "The type of the image is not np.uint8"
            assert np.all(space.low >= 0) and np.all(space.high <= 255), "The value of the image is not in [0, 255]"


def test_return_data():
    # Capture warning from making environment
    with pytest.warns(None) as warnings:
        env = gym.make("snake:snake-v0")

    # Check return data
    action_space = env.action_space

    obs = env.reset()
    assert isinstance(obs, dict), "The observation should return `dict`"

    action = action_space.sample()
    data = env.step(action)

    assert (len(data) == 4), "The `step()` method must return four values: obs, reward, done, info"
