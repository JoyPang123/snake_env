import os
import torch

from DQN_algo.model import DQN
from A2C_algo.model import ActorCritic

NUM_ACTIONS = 4

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def test_dqn():
    random_tensor = torch.rand([1, 3, 64, 64])
    model = DQN(NUM_ACTIONS)
    model.eval()
    q_value = model(random_tensor)

    assert q_value.shape == (1, NUM_ACTIONS)


def test_actor_critic():
    random_tensor = torch.rand([1, 3, 64, 64])
    model = ActorCritic(NUM_ACTIONS)
    model.eval()
    actor, critic = model(random_tensor)

    assert actor.shape == (1, NUM_ACTIONS) and critic.shape == (1, 1)
