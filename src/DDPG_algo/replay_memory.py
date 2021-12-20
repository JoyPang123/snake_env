from collections import deque
import random


class ReplayMemory:
    def __init__(self, max_len):
        self.replay = deque(maxlen=max_len)

    def store_experience(self, state, reward,
                         action, next_state,
                         done):
        self.replay.append([state, reward, action, next_state, done])

    def size(self):
        return len(self.replay)

    def sample(self, batch_size):
        if len(self.replay) < batch_size:
            return None

        return random.sample(self.replay, k=batch_size)
