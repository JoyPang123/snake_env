import os
import torch

from src.DQN_algo import DQN
from src.A2C_algo.model import ActorCritic as AC
from src.PPO_algo.model import ActorCritic as PPO_AC

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
    model = AC(NUM_ACTIONS)
    model.eval()
    actor, critic = model(random_tensor)

    assert actor.shape == (1, NUM_ACTIONS) and critic.shape == (1, 1)


def test_ppo():
    random_tensor = torch.rand([1, 3, 64, 64])
    model = PPO_AC(NUM_ACTIONS)
    model.eval()
    act_prob, act_logprob = model.act(random_tensor)
    act_logprobs, state_values, dist_entropy = model.evaluate(random_tensor, torch.tensor([0]))

    assert act_prob.shape == (1,)
    assert act_logprob.shape == (1,)
    assert act_logprobs.shape == (1,)
    assert state_values.shape == (1, 1)
    assert dist_entropy.shape == (1,)
